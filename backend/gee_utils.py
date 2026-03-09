from __future__ import annotations

import calendar
import logging
from datetime import date, timedelta

import ee

LOGGER = logging.getLogger("vegetation-api")

REQUIRED_9_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A"]
WEB_MERCATOR = "EPSG:3857"
CLOUD_BIT_MASK = 1 << 10
CIRRUS_BIT_MASK = 1 << 11
CLOUD_SHADOW_SCL_CLASS = 3
CLOUD_SCL_CLASSES = [8, 9, 10]


def init_ee() -> None:
    # Earth Engine is initialized once at app startup.
    ee.Initialize(project="trusty-entity-462211-b8")
    # Increase EE client-side RPC deadline (milliseconds) for heavy AOI/year requests.
    ee.data.setDeadline(300000)


def _safe_number(value: object, decimals: int = 6) -> float:
    if value is None:
        return 0.0
    return round(float(value), decimals)


def _mask_clouds_and_shadows(img: ee.Image) -> ee.Image:
    # QA60 masks opaque clouds/cirrus, and SCL masks cloud shadows and cloud classes.
    qa60 = img.select("QA60")
    qa_clear = qa60.bitwiseAnd(CLOUD_BIT_MASK).eq(0).And(qa60.bitwiseAnd(CIRRUS_BIT_MASK).eq(0))

    scl = img.select("SCL")
    scl_clear = (
        scl.neq(CLOUD_SHADOW_SCL_CLASS)
        .And(scl.neq(CLOUD_SCL_CLASSES[0]))
        .And(scl.neq(CLOUD_SCL_CLASSES[1]))
        .And(scl.neq(CLOUD_SCL_CLASSES[2]))
    )

    return img.updateMask(qa_clear.And(scl_clear)).copyProperties(img, img.propertyNames())


def _valid_fraction_for_b4(image: ee.Image, aoi: ee.Geometry) -> float:
    valid_fraction_raw = (
        image.select("B4")
        .mask()
        .rename("valid")
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=1_000_000_000,
            bestEffort=True,
        )
        .get("valid")
    )
    return _safe_number(valid_fraction_raw.getInfo(), 6)


def _build_composite(
    base_collection: ee.ImageCollection,
    aoi: ee.Geometry,
    start_date: date,
    end_date: date,
    apply_mask: bool,
) -> tuple[ee.Image | None, int, float]:
    collection = base_collection.filterDate(start_date.isoformat(), end_date.isoformat())
    if apply_mask:
        collection = collection.map(_mask_clouds_and_shadows)

    collection = collection.map(lambda img: img.select(REQUIRED_9_BANDS).toFloat())
    image_count = int(collection.size().getInfo())
    if image_count == 0:
        return None, 0, 0.0

    image = ee.Image(collection.median()).clip(aoi)
    if apply_mask:
        return image, image_count, _valid_fraction_for_b4(image, aoi)
    return image, image_count, 1.0


def process_year_with_visuals(
    aoi: ee.Geometry,
    year: int,
    month: int,
    day: int,
) -> tuple[dict[str, float | int] | None, dict[str, str] | None, list[str]]:
    debug: list[str] = []

    try:
        month_start_date = date(year, month, 1)
        if month == 12:
            month_end_date = date(year + 1, 1, 1)
        else:
            month_end_date = date(year, month + 1, 1)

        month_scene_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(month_start_date.isoformat(), month_end_date.isoformat())
            # Keep more scenes and evaluate with pixel-level cloud/shadow masks.
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        )

        month_image_count = int(month_scene_collection.size().getInfo())
        msg_count = f"Year {year}, month {month:02d} - scenes found: {month_image_count}"
        debug.append(msg_count)
        LOGGER.info(msg_count)

        if month_image_count == 0:
            warn = f"Year {year}, month {month:02d} - skipped (no images found after cloud filtering)."
            debug.append(warn)
            LOGGER.warning(warn)
            return None, None, debug

        month_days = calendar.monthrange(year, month)[1]
        start_day = min(max(int(day), 1), month_days)
        selected_image: ee.Image | None = None
        selected_mode = ""
        selected_valid_fraction = 0.0
        month_masked_image, masked_count, masked_valid_fraction = _build_composite(
            month_scene_collection, aoi, month_start_date, month_end_date, apply_mask=True
        )
        debug.append(
            f"Year {year}, month {month:02d} - masked monthly pool scenes: {masked_count}, valid coverage: {round(masked_valid_fraction * 100.0, 2)}%"
        )

        # Preferred behavior: same day each year; if unavailable, move forward day by day.
        for candidate_day in range(start_day, month_days + 1):
            day_start = date(year, month, candidate_day)
            day_end = day_start + timedelta(days=1)
            day_image, day_count, day_valid_fraction = _build_composite(
                month_scene_collection, aoi, day_start, day_end, apply_mask=True
            )
            debug.append(
                f"Year {year}, month {month:02d}, day {candidate_day:02d} - masked scenes: {day_count}, valid coverage: {round(day_valid_fraction * 100.0, 2)}%"
            )
            if day_image is not None and day_valid_fraction > 0.0:
                selected_image = day_image
                selected_mode = f"masked day+ ({candidate_day:02d})"
                selected_valid_fraction = day_valid_fraction
                break

        if selected_image is None and month_masked_image is not None and masked_valid_fraction > 0.0:
            selected_image = month_masked_image
            selected_mode = "masked monthly fallback"
            selected_valid_fraction = masked_valid_fraction

        if selected_image is None:
            warn = f"Year {year}, month {month:02d} - skipped (no valid cloud-masked imagery found)."
            debug.append(warn)
            LOGGER.warning(warn)
            return None, None, debug

        # Gap-fill day-level masked composite using the full-month masked composite.
        if selected_mode.startswith("masked day+") and month_masked_image is not None:
            selected_image = selected_image.unmask(month_masked_image).clip(aoi)
            selected_valid_fraction = _valid_fraction_for_b4(selected_image, aoi)
            selected_mode = f"{selected_mode} + masked monthly fill"

        debug.append(
            f"Year {year}, month {month:02d} - selected source: {selected_mode}, valid coverage: {round(selected_valid_fraction * 100.0, 2)}%"
        )
        image = selected_image

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

        debug.append(f"Year {year}, month {month:02d} - NDVI mean: {ndvi_mean}")
        debug.append(f"Year {year}, month {month:02d} - Vegetation %: {veg_percent}")
        LOGGER.info("Year %s month %02d - NDVI mean: %s", year, month, ndvi_mean)
        LOGGER.info("Year %s month %02d - Vegetation %%: %s", year, month, veg_percent)

        # Render in Web Mercator and stream as EE tiles.
        # Keep masked/no-data pixels transparent so they are not visualized as low-NDVI red.
        ndvi_vis = (
            ndvi.visualize(min=-0.2, max=0.8, palette=["#d73027", "#fee08b", "#1a9850"])
            .reproject(crs=WEB_MERCATOR, scale=30)
        )
        map_id = ndvi_vis.getMapId({})
        ndvi_tile_url = map_id["tile_fetcher"].url_format

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
            "ndviTileUrl": ndvi_tile_url,
        }

        return feature, maps, debug
    except Exception as exc:
        warn = f"Year {year}, month {month:02d} - Earth Engine failure: {exc}"
        debug.append(warn)
        LOGGER.warning(warn)
        return None, None, debug
