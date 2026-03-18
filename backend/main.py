from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

import ee
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from gee_utils import init_ee, process_year_with_visuals
from model_adapter import predict_risk_with_debug

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger("vegetation-api")

MAX_YEAR_IMAGES = 10

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_ee()
    LOGGER.info("Earth Engine initialized at startup with project trusty-entity-462211-b8")
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


@app.post("/analyze", responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def analyze(payload: AnalyzeRequest) -> dict[str, object]:
    debug: list[str] = []

    if payload.endYear <= payload.startYear:
        raise HTTPException(status_code=400, detail="endYear must be greater than startYear")

    # End year is exclusive: 2015 -> 2025 means 10 yearly images (2015..2024).
    requested_years = list(range(payload.startYear, payload.endYear))
    if len(requested_years) > MAX_YEAR_IMAGES:
        raise HTTPException(status_code=400, detail=f"Date range too large. Use at most {MAX_YEAR_IMAGES} years.")

    # AOI is created directly from the user-selected map rectangle.
    aoi = ee.Geometry.Rectangle(payload.bbox)

    features: list[dict[str, float | int]] = []
    years: list[int] = []
    ndvi_series: list[float] = []
    ndwi_series: list[float] = []
    evi_series: list[float] = []
    ndmi_series: list[float] = []
    savi_series: list[float] = []
    vegetation_series: list[float] = []
    maps: dict[str, dict[str, str]] = {}

    for year in requested_years:
        feature, map_urls, year_debug = process_year_with_visuals(aoi, year, payload.month, payload.day)
        debug.extend(year_debug)

        if feature is None or map_urls is None:
            continue

        features.append(feature)
        years.append(int(feature["year"]))
        ndvi_series.append(float(feature["ndvi"]))
        ndwi_series.append(float(feature["ndwi"]))
        evi_series.append(float(feature["evi"]))
        ndmi_series.append(float(feature["ndmi"]))
        savi_series.append(float(feature["savi"]))
        vegetation_series.append(float(feature["vegetationPercent"]))
        maps[str(year)] = map_urls

    if not features:
        raise HTTPException(status_code=400, detail="No valid yearly features were produced for the selected AOI/time range")

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
        "features": features,
        "maps": maps,
        "risk": risk,
        "debug": debug,
    }
