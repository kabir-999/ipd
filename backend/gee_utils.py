from __future__ import annotations

import logging

import ee

LOGGER = logging.getLogger("vegetation-api")

PROCESS_BANDS = ["B2", "B3", "B4", "B8", "B8A", "B11"]
WEB_MERCATOR = "EPSG:3857"
CLOUD_BIT_MASK = 1 << 10
CIRRUS_BIT_MASK = 1 << 11
CLOUD_SHADOW_SCL_CLASS = 3
CLOUD_SCL_CLASSES = [8, 9, 10]
PRIMARY_SCENE_LIMIT = 3
FALLBACK_SCENE_LIMIT = 3
MIN_PRIMARY_SCENES = 1
MIN_VALID_COVERAGE = 0.35
NDVI_VEG_THRESHOLD = 0.35
EVI_VEG_THRESHOLD = 0.20
SAVI_VEG_THRESHOLD = 0.25
NDMI_VEG_THRESHOLD = 0.00
NDWI_VEG_THRESHOLD = -0.05


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


def _composite_from_collection(collection: ee.ImageCollection, aoi: ee.Geometry) -> tuple[ee.Image, float]:
    masked_collection = (
        collection
        .map(_mask_clouds_and_shadows)
        .map(lambda img: img.select(PROCESS_BANDS).toFloat())
    )
    image = ee.Image(masked_collection.median()).clip(aoi)
    return image, _valid_fraction_for_b4(image, aoi)


def process_year_with_visuals(
    aoi: ee.Geometry,
    year: int,
    month: int,
    day: int,
) -> tuple[dict[str, float | int] | None, dict[str, str] | None, list[str]]:
    debug: list[str] = []

    try:
        month_start = f"{year}-{month:02d}-01"
        if month == 12:
            month_end = f"{year + 1}-01-01"
        else:
            month_end = f"{year}-{month + 1:02d}-01"

        month_scene_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(month_start, month_end)
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

        # Primary pass: same month/day window with limited scenes for speed + stability.
        day_int = min(max(int(day), 1), 28)
        window_start = f"{year}-{month:02d}-{day_int:02d}"
        if month == 12:
            window_end = f"{year + 1}-01-01"
        else:
            window_end = f"{year}-{month + 1:02d}-01"

        window_collection = month_scene_collection.filterDate(window_start, window_end).sort("CLOUDY_PIXEL_PERCENTAGE")
        window_count = int(window_collection.size().getInfo())
        debug.append(f"Year {year}, month {month:02d} - day-window scenes found: {window_count}")

        if window_count >= MIN_PRIMARY_SCENES:
            primary_collection = window_collection.limit(PRIMARY_SCENE_LIMIT)
            source_label = "day-window composite"
        else:
            primary_collection = month_scene_collection.sort("CLOUDY_PIXEL_PERCENTAGE").limit(PRIMARY_SCENE_LIMIT)
            source_label = "month composite fallback"

        used_scene_count = int(primary_collection.size().getInfo())
        debug.append(f"Year {year}, month {month:02d} - scenes used for analysis: {used_scene_count} ({source_label})")
        if used_scene_count == 0:
            warn = f"Year {year}, month {month:02d} - skipped (no scenes available for analysis)."
            debug.append(warn)
            LOGGER.warning(warn)
            return None, None, debug

        image, valid_fraction = _composite_from_collection(primary_collection, aoi)
        debug.append(f"Year {year}, month {month:02d} - valid pixel coverage: {round(valid_fraction * 100.0, 2)}%")
        selected_source = source_label

        # Secondary pass: if coverage is poor, use a broader monthly pool.
        if valid_fraction < MIN_VALID_COVERAGE:
            fallback_collection = month_scene_collection.sort("CLOUDY_PIXEL_PERCENTAGE").limit(FALLBACK_SCENE_LIMIT)
            fallback_count = int(fallback_collection.size().getInfo())
            debug.append(f"Year {year}, month {month:02d} - fallback scenes used: {fallback_count}")
            if fallback_count > 0:
                fallback_image, fallback_valid_fraction = _composite_from_collection(fallback_collection, aoi)
                if fallback_valid_fraction > valid_fraction:
                    image = fallback_image
                    valid_fraction = fallback_valid_fraction
                    selected_source = "broader monthly fallback"
                    debug.append(
                        f"Year {year}, month {month:02d} - upgraded to fallback coverage: {round(valid_fraction * 100.0, 2)}%"
                    )

        if valid_fraction == 0.0:
            warn = f"Year {year}, month {month:02d} - skipped (no valid cloud-masked pixels in selected/fallback scenes)."
            debug.append(warn)
            LOGGER.warning(warn)
            return None, None, debug

        debug.append(f"Year {year}, month {month:02d} - selected source: {selected_source}")

        ndvi = image.normalizedDifference(["B8", "B4"]).rename("ndvi")
        ndwi = image.normalizedDifference(["B3", "B8"]).rename("ndwi")
        ndmi = image.normalizedDifference(["B8", "B11"]).rename("ndmi")
        savi = image.expression(
            "1.5 * ((nir - red) / (nir + red + 0.5))",
            {"nir": image.select("B8"), "red": image.select("B4")},
        ).rename("savi")
        evi = image.expression(
            "2.5 * ((nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0))",
            {"nir": image.select("B8"), "red": image.select("B4"), "blue": image.select("B2")},
        ).rename("evi")

        veg_score = (
            ndvi.gt(NDVI_VEG_THRESHOLD)
            .add(evi.gt(EVI_VEG_THRESHOLD))
            .add(savi.gt(SAVI_VEG_THRESHOLD))
            .add(ndmi.gt(NDMI_VEG_THRESHOLD))
            .add(ndwi.gt(NDWI_VEG_THRESHOLD))
        )
        veg_mask = veg_score.gte(3).rename("veg")

        stats_raw = ndvi.addBands([ndwi, evi, ndmi, savi, veg_mask]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=1_000_000_000,
            bestEffort=True,
        ).getInfo()

        ndvi_mean = _safe_number(stats_raw.get("ndvi"), 6)
        ndwi_mean = _safe_number(stats_raw.get("ndwi"), 6)
        evi_mean = _safe_number(stats_raw.get("evi"), 6)
        ndmi_mean = _safe_number(stats_raw.get("ndmi"), 6)
        savi_mean = _safe_number(stats_raw.get("savi"), 6)
        veg_percent = round(_safe_number(stats_raw.get("veg"), 6) * 100.0, 4)

        debug.append(f"Year {year}, month {month:02d} - NDVI mean: {ndvi_mean}")
        debug.append(f"Year {year}, month {month:02d} - NDWI mean: {ndwi_mean}")
        debug.append(f"Year {year}, month {month:02d} - EVI mean: {evi_mean}")
        debug.append(f"Year {year}, month {month:02d} - NDMI mean: {ndmi_mean}")
        debug.append(f"Year {year}, month {month:02d} - SAVI mean: {savi_mean}")
        debug.append(f"Year {year}, month {month:02d} - Vegetation %: {veg_percent}")
        LOGGER.info("Year %s month %02d - NDVI mean: %s", year, month, ndvi_mean)
        LOGGER.info("Year %s month %02d - NDWI mean: %s", year, month, ndwi_mean)
        LOGGER.info("Year %s month %02d - EVI mean: %s", year, month, evi_mean)
        LOGGER.info("Year %s month %02d - NDMI mean: %s", year, month, ndmi_mean)
        LOGGER.info("Year %s month %02d - SAVI mean: %s", year, month, savi_mean)
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
            "ndwi": ndwi_mean,
            "savi": savi_mean,
            "evi": evi_mean,
            "ndmi": ndmi_mean,
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
