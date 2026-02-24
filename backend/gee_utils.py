from __future__ import annotations

import logging
import time
from pathlib import Path

import ee
import requests

LOGGER = logging.getLogger("vegetation-api")

S2_BASE_BANDS = ["B2", "B3", "B4", "B8"]
REQUIRED_WIDTH = 10534
REQUIRED_HEIGHT = 7778
REQUIRED_9_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A"]
THUMB_MAX_DIM = 1024


def init_ee() -> None:
    # Earth Engine is initialized once at app startup.
    ee.Initialize(project="trusty-entity-462211-b8")
    # Increase EE client-side RPC deadline (milliseconds) for heavy AOI/year requests.
    ee.data.setDeadline(300000)


def _safe_number(value: object, decimals: int = 6) -> float:
    if value is None:
        return 0.0
    return round(float(value), decimals)


def _download_thumb(url: str, path: Path, http: requests.Session) -> None:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = http.get(url, timeout=(20, 240))
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            return
        except requests.RequestException as exc:
            last_error = exc
            LOGGER.warning("Thumbnail download failed (attempt %s/3): %s", attempt, exc)
            time.sleep(attempt * 1.5)

    raise RuntimeError(f"Thumbnail download failed after retries: {last_error}")


def process_year_with_visuals(
    aoi: ee.Geometry,
    bbox: list[float],
    year: int,
    generated_dir: Path,
    http: requests.Session,
) -> tuple[dict[str, float | int] | None, dict[str, str] | None, list[str]]:
    debug: list[str] = []

    try:
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 35))
            # Enforce exactly 9 bands in the processing input.
            .map(lambda img: img.select(REQUIRED_9_BANDS).toFloat())
        )

        image_count = int(collection.size().getInfo())
        msg_count = f"Year {year} - images found: {image_count}"
        debug.append(msg_count)
        LOGGER.info(msg_count)

        if image_count == 0:
            warn = f"Year {year} - skipped (no images found after cloud filtering)."
            debug.append(warn)
            LOGGER.warning(warn)
            return None, None, debug

        def score_image(img: ee.Image) -> ee.Image:
            overlap_area = img.geometry().intersection(aoi, ee.ErrorMargin(1)).area(1)
            cloud = ee.Number(img.get("CLOUDY_PIXEL_PERCENTAGE"))
            # Prefer larger AOI overlap first, then lower cloud.
            score = ee.Number(overlap_area).subtract(cloud.multiply(1e8))
            return img.set("selectionScore", score)

        # Use exactly one image per year: best AOI overlap + cloud score.
        image = ee.Image(collection.map(score_image).sort("selectionScore", False).first()).clip(aoi)
        debug.append(f"Year {year} - enforced input shape target: {REQUIRED_HEIGHT}x{REQUIRED_WIDTH} with 9 bands")

        ndvi = image.normalizedDifference(["B8", "B4"]).rename("ndvi")
        veg_mask = ndvi.gt(0.3).rename("veg")

        ndvi_mean_raw = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=1_000_000_000,
            bestEffort=True,
        ).get("ndvi")

        veg_mean_raw = veg_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=1_000_000_000,
            bestEffort=True,
        ).get("veg")

        ndvi_mean = _safe_number(ndvi_mean_raw.getInfo(), 6)
        veg_percent = round(_safe_number(veg_mean_raw.getInfo(), 6) * 100.0, 4)

        debug.append(f"Year {year} - NDVI mean: {ndvi_mean}")
        debug.append(f"Year {year} - Vegetation %: {veg_percent}")
        LOGGER.info("Year %s - NDVI mean: %s", year, ndvi_mean)
        LOGGER.info("Year %s - Vegetation %%: %s", year, veg_percent)

        # Visualization generation happens here using one Earth Engine thumbnail per year.
        ndvi_vis = ndvi.unmask(-0.2).visualize(min=-0.2, max=0.8, palette=["#d73027", "#fee08b", "#1a9850"])

        thumb_region = {
            "type": "Polygon",
            "coordinates": [[
                [bbox[0], bbox[1]],
                [bbox[0], bbox[3]],
                [bbox[2], bbox[3]],
                [bbox[2], bbox[1]],
                [bbox[0], bbox[1]],
            ]],
        }

        # Thumbnails are for UI display only; keep size below EE request-size limits.
        aspect = REQUIRED_WIDTH / REQUIRED_HEIGHT
        thumb_w = THUMB_MAX_DIM
        thumb_h = max(1, int(round(THUMB_MAX_DIM / aspect)))

        ndvi_url = ndvi_vis.getThumbURL(
            {
                "region": thumb_region,
                "dimensions": f"{thumb_w}x{thumb_h}",
                "crs": "EPSG:4326",
                "format": "png",
            }
        )

        ndvi_path = generated_dir / f"ndvi_{year}.png"
        _download_thumb(ndvi_url, ndvi_path, http)

        feature = {
            "year": year,
            "ndvi": ndvi_mean,
            "vegetationPercent": veg_percent,
            "ndwi": 0.0,
            "savi": 0.0,
            "evi": 0.0,
            "ndmi": 0.0,
            "nbr": 0.0,
        }

        maps = {
            "ndvi": f"/generated/{ndvi_path.name}",
        }

        return feature, maps, debug
    except Exception as exc:
        warn = f"Year {year} - Earth Engine failure: {exc}"
        debug.append(warn)
        LOGGER.warning(warn)
        return None, None, debug
