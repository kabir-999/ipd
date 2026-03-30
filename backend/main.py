from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from gee_utils import process
from model_adapter import predict_risk_with_debug

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger("vegetation-api")

MAX_YEAR_IMAGES = 10


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield


app = FastAPI(title="Vegetation Monitoring API", version="5.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ErrorResponse(BaseModel):
    detail: str


class AnalyzeRequest(BaseModel):
    bbox: list[float] = Field(min_length=4, max_length=4)
    startYear: int
    endYear: int
    month: int
    day: int = 1

    @field_validator("startYear", "endYear")
    @classmethod
    def validate_year(cls, value: int) -> int:
        current_year = datetime.utcnow().year
        if value < 2015 or value > current_year:
            raise ValueError(f"Year must be between 2015 and {current_year}")
        return value

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("bbox must be [minLon, minLat, maxLon, maxLat]")

        min_lon, min_lat, max_lon, max_lat = value
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("Invalid bbox ordering")

        if min_lon < -180 or max_lon > 180 or min_lat < -90 or max_lat > 90:
            raise ValueError("bbox values are out of geographic range")

        return value

    @field_validator("month")
    @classmethod
    def validate_month(cls, value: int) -> int:
        if value < 1 or value > 12:
            raise ValueError("month must be between 1 and 12")
        return value

    @field_validator("day")
    @classmethod
    def validate_day(cls, value: int) -> int:
        if value < 1 or value > 31:
            raise ValueError("day must be between 1 and 31")
        return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled backend error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {exc}"})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "vegetation-monitoring"}


def _feature_from_stats(year: int, stats: dict[str, object]) -> dict[str, float | int | str]:
    return {
        "year": year,
        "source": str(stats.get("source", "")),
        "ndvi": float(stats.get("ndvi", 0.0)),
        "ndwi": float(stats.get("ndwi", 0.0)),
        "evi": float(stats.get("evi", 0.0)),
        "ndmi": float(stats.get("ndmi", 0.0)),
        "savi": float(stats.get("savi", 0.0)),
        "nbr": float(stats.get("nbr", 0.0)),
        "forestAreaSqm": float(stats.get("forestAreaSqm", 0.0)),
        "forestPercent": float(stats.get("forestPercent", 0.0)),
        "vegetationPercent": float(stats.get("vegetationPercent", 0.0)),
        "nonVegetationPercent": float(stats.get("nonVegetationPercent", 0.0)),
        "sparseVegetationPercent": float(stats.get("sparseVegetationPercent", 0.0)),
        "denseVegetationPercent": float(stats.get("denseVegetationPercent", 0.0)),
        "validCoveragePercent": float(stats.get("validCoveragePercent", 0.0)),
    }


def _debug_from_stats(year: int, stats: dict[str, object]) -> list[str]:
    return [
        f"Year {year} - source: {stats.get('source', 'unknown')}",
        (
            f"Year {year} - window: {stats.get('windowStart', 'n/a')} to "
            f"{stats.get('windowEnd', 'n/a')} ({stats.get('sceneCount', 0)} scenes)"
        ),
        f"Year {year} - valid coverage: {stats.get('validCoveragePercent', 0)}%",
        f"Year {year} - NDVI mean: {stats.get('ndvi', 0)}",
        f"Year {year} - forest cover (tree cover > 30 and NDVI > 0.3): {stats.get('forestPercent', 0)}%",
        f"Year {year} - vegetation cover (>0.3 NDVI): {stats.get('vegetationPercent', 0)}%",
    ]


def _apply_deforestation_signals(features: list[dict[str, float | int | str]], debug: list[str]) -> float:
    if not features:
        return 0.0

    baseline_forest_percent = float(features[0].get("forestPercent", 0.0))
    previous_forest_percent: float | None = None

    for feature in features:
        year = int(feature["year"])
        forest_percent = float(feature.get("forestPercent", 0.0))
        if previous_forest_percent is None:
            forest_change_percent = 0.0
        else:
            forest_change_percent = round(forest_percent - previous_forest_percent, 4)

        total_forest_loss = max(0.0, baseline_forest_percent - forest_percent)
        yearly_forest_loss = max(0.0, -forest_change_percent)
        deforestation_risk = round(min(1.0, max(total_forest_loss, yearly_forest_loss) / 100.0), 4)

        feature["forestChangePercent"] = forest_change_percent
        feature["deforestationRisk"] = deforestation_risk
        debug.append(
            f"Year {year} - forest change vs previous year: {forest_change_percent}% | "
            f"deforestation risk: {deforestation_risk}"
        )
        previous_forest_percent = forest_percent

    return round(float(features[-1].get("deforestationRisk", 0.0)), 4)


@app.post("/analyze", responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
def analyze(payload: AnalyzeRequest) -> dict[str, object]:
    debug: list[str] = []

    if payload.endYear <= payload.startYear:
        raise HTTPException(status_code=400, detail="endYear must be greater than startYear")

    requested_years = list(range(payload.startYear, payload.endYear))
    if len(requested_years) > MAX_YEAR_IMAGES:
        raise HTTPException(status_code=400, detail=f"Date range too large. Use at most {MAX_YEAR_IMAGES} years.")

    features: list[dict[str, float | int | str]] = []
    years: list[int] = []
    ndvi_series: list[float] = []
    ndwi_series: list[float] = []
    evi_series: list[float] = []
    ndmi_series: list[float] = []
    savi_series: list[float] = []
    vegetation_series: list[float] = []
    forest_series: list[float] = []
    forest_area_series: list[float] = []
    maps: dict[str, dict[str, str]] = {}

    for year in requested_years:
        try:
            result = process(payload.bbox, year, payload.month, payload.day)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            debug.append(f"Year {year} - {exc}")
            continue

        stats = result["stats"]
        map_urls = result["maps"]
        feature = _feature_from_stats(year, stats)

        features.append(feature)
        years.append(year)
        ndvi_series.append(float(feature["ndvi"]))
        ndwi_series.append(float(feature["ndwi"]))
        evi_series.append(float(feature["evi"]))
        ndmi_series.append(float(feature["ndmi"]))
        savi_series.append(float(feature["savi"]))
        vegetation_series.append(float(feature["vegetationPercent"]))
        forest_series.append(float(feature["forestPercent"]))
        forest_area_series.append(float(feature["forestAreaSqm"]))
        maps[str(year)] = map_urls
        debug.extend(_debug_from_stats(year, stats))

    if not features:
        raise HTTPException(status_code=400, detail="No valid yearly features were produced for the selected AOI/time range")

    deforestation_risk = _apply_deforestation_signals(features, debug)
    risk, lstm_debug = predict_risk_with_debug(features)
    debug.extend(lstm_debug)

    return {
        "years": years,
        "ndvi": ndvi_series,
        "ndwi": ndwi_series,
        "evi": evi_series,
        "ndmi": ndmi_series,
        "savi": savi_series,
        "vegetation": vegetation_series,
        "forest": forest_series,
        "forestAreaSqm": forest_area_series,
        "features": features,
        "maps": maps,
        "risk": risk,
        "deforestationRisk": deforestation_risk,
        "debug": debug,
    }
