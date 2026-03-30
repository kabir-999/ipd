from __future__ import annotations

import calendar
import logging
from datetime import date, timedelta
from typing import Any

import ee

LOGGER = logging.getLogger("vegetation-api")

EE_PROJECT = "trusty-entity-462211-b8"
S2_COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"
HANSEN_COLLECTION_ID = "UMD/hansen/global_forest_change_2023_v1_11"
PROCESS_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
ANALYSIS_SCALE = 10
MAX_PIXELS = 1_000_000_000
PRIMARY_WINDOW_DAYS = 45
FALLBACK_WINDOW_DAYS = 90
MAX_SCENE_CLOUD_PERCENT = 80
WEB_MERCATOR = "EPSG:3857"

NON_VEGETATION_MAX_NDVI = 0.2
VEGETATION_MASK_THRESHOLD = 0.3
DENSE_VEGETATION_MIN_NDVI = 0.5
TREE_COVER_THRESHOLD = 30

_EE_INITIALIZED = False


def init_ee() -> None:
    global _EE_INITIALIZED

    if _EE_INITIALIZED:
        return

    try:
        ee.Initialize(project=EE_PROJECT)
        ee.data.setDeadline(300000)
        _EE_INITIALIZED = True
    except Exception as exc:
        raise RuntimeError(f"Earth Engine initialization failed: {exc}") from exc


def _safe_number(value: Any, decimals: int = 6) -> float:
    if value is None:
        return 0.0
    return round(float(value), decimals)


def _clamp_day(year: int, month: int, day: int) -> int:
    return min(max(int(day), 1), calendar.monthrange(year, month)[1])


def _build_window(year: int, month: int, day: int, window_days: int) -> tuple[str, str]:
    selected_date = date(year, month, _clamp_day(year, month, day))
    start_date = selected_date - timedelta(days=window_days // 2)
    end_date = start_date + timedelta(days=window_days)
    return start_date.isoformat(), end_date.isoformat()


def _mask_sentinel2_clouds(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    clear_mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))

    return (
        image.updateMask(clear_mask)
        .select(PROCESS_BANDS)
        .multiply(0.0001)
        .toFloat()
        .copyProperties(image, image.propertyNames())
    )


def _normalize_aoi(aoi: ee.Geometry | list[float] | tuple[float, float, float, float]) -> ee.Geometry:
    if isinstance(aoi, (list, tuple)):
        if len(aoi) != 4:
            raise ValueError("bbox must contain [minLon, minLat, maxLon, maxLat].")
        min_lon, min_lat, max_lon, max_lat = [float(value) for value in aoi]
        return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    return aoi


def _get_collection(aoi: ee.Geometry, start_date: str, end_date: str) -> ee.ImageCollection:
    return (
        ee.ImageCollection(S2_COLLECTION_ID)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_SCENE_CLOUD_PERCENT))
        .map(_mask_sentinel2_clouds)
    )


def _valid_pixel_fraction(image: ee.Image, aoi: ee.Geometry) -> float:
    raw = (
        image.select("B4")
        .mask()
        .rename("valid")
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=ANALYSIS_SCALE,
            maxPixels=MAX_PIXELS,
            bestEffort=True,
        )
        .get("valid")
    )
    return _safe_number(raw.getInfo())


def _get_composite(aoi: ee.Geometry, year: int, month: int, day: int) -> tuple[ee.Image, dict[str, str | int | float]]:
    windows = [
        ("45-day median composite", *_build_window(year, month, day, PRIMARY_WINDOW_DAYS)),
        ("90-day median composite fallback", *_build_window(year, month, day, FALLBACK_WINDOW_DAYS)),
    ]

    best_image: ee.Image | None = None
    best_metadata: dict[str, str | int | float] | None = None
    best_valid_fraction = -1.0

    for source, start_date, end_date in windows:
        collection = _get_collection(aoi, start_date, end_date)
        scene_count = int(collection.size().getInfo())
        if scene_count == 0:
            continue

        composite = ee.Image(collection.median()).clip(aoi)
        valid_fraction = _valid_pixel_fraction(composite, aoi)
        if valid_fraction <= best_valid_fraction:
            continue

        best_image = composite
        best_valid_fraction = valid_fraction
        best_metadata = {
            "source": source,
            "sceneCount": scene_count,
            "windowStart": start_date,
            "windowEnd": end_date,
            "validCoveragePercent": round(valid_fraction * 100.0, 4),
        }

    if best_image is None or best_metadata is None or best_valid_fraction <= 0:
        raise ValueError("No valid cloud-masked Sentinel-2 imagery was available for the selected year.")

    return best_image, best_metadata


def _classify_ndvi(ndvi: ee.Image) -> tuple[ee.Image, ee.Image]:
    classified = (
        ee.Image(0)
        .where(ndvi.gte(NON_VEGETATION_MAX_NDVI).And(ndvi.lte(DENSE_VEGETATION_MIN_NDVI)), 1)
        .where(ndvi.gt(DENSE_VEGETATION_MIN_NDVI), 2)
        .updateMask(ndvi.mask())
        .rename("ndviClass")
    )
    vegetation_mask = ndvi.gt(VEGETATION_MASK_THRESHOLD).updateMask(ndvi.mask()).rename("vegetationMask")
    return classified, vegetation_mask


def _get_tree_mask(aoi: ee.Geometry) -> ee.Image:
    return (
        ee.Image(HANSEN_COLLECTION_ID)
        .select("treecover2000")
        .gt(TREE_COVER_THRESHOLD)
        .clip(aoi)
        .rename("treeMask")
    )


def _compute_index_stats(composite: ee.Image, ndvi: ee.Image, aoi: ee.Geometry) -> dict[str, float]:
    ndwi = composite.normalizedDifference(["B3", "B8"]).rename("ndwi")
    ndmi = composite.normalizedDifference(["B8", "B11"]).rename("ndmi")
    nbr = composite.normalizedDifference(["B8", "B12"]).rename("nbr")
    savi = composite.expression(
        "1.5 * ((nir - red) / (nir + red + 0.5))",
        {"nir": composite.select("B8"), "red": composite.select("B4")},
    ).rename("savi")
    evi = composite.expression(
        "2.5 * ((nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0))",
        {"nir": composite.select("B8"), "red": composite.select("B4"), "blue": composite.select("B2")},
    ).rename("evi")

    stats = (
        ndvi.addBands([ndwi, evi, ndmi, savi, nbr])
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=ANALYSIS_SCALE,
            maxPixels=MAX_PIXELS,
            bestEffort=True,
        )
        .getInfo()
    )

    return {
        "ndvi": _safe_number(stats.get("ndvi")),
        "ndwi": _safe_number(stats.get("ndwi")),
        "evi": _safe_number(stats.get("evi")),
        "ndmi": _safe_number(stats.get("ndmi")),
        "savi": _safe_number(stats.get("savi")),
        "nbr": _safe_number(stats.get("nbr")),
    }


def _compute_area_stats(
    ndvi: ee.Image,
    classified: ee.Image,
    vegetation_mask: ee.Image,
    forest_mask: ee.Image,
    aoi: ee.Geometry,
) -> dict[str, float]:
    pixel_area = ee.Image.pixelArea()
    area_stats = (
        pixel_area.updateMask(ndvi.mask())
        .rename("validArea")
        .addBands(pixel_area.updateMask(vegetation_mask).rename("vegetationArea"))
        .addBands(pixel_area.updateMask(forest_mask).rename("forestArea"))
        .addBands(pixel_area.updateMask(classified.eq(0)).rename("nonVegetationArea"))
        .addBands(pixel_area.updateMask(classified.eq(1)).rename("sparseVegetationArea"))
        .addBands(pixel_area.updateMask(classified.eq(2)).rename("denseVegetationArea"))
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=ANALYSIS_SCALE,
            maxPixels=MAX_PIXELS,
            bestEffort=True,
        )
        .getInfo()
    )

    valid_area = _safe_number(area_stats.get("validArea"), 2)
    denominator = valid_area if valid_area > 0 else 1.0

    vegetation_area = _safe_number(area_stats.get("vegetationArea"), 2)
    forest_area = _safe_number(area_stats.get("forestArea"), 2)
    non_vegetation_area = _safe_number(area_stats.get("nonVegetationArea"), 2)
    sparse_vegetation_area = _safe_number(area_stats.get("sparseVegetationArea"), 2)
    dense_vegetation_area = _safe_number(area_stats.get("denseVegetationArea"), 2)

    return {
        "forestAreaSqm": forest_area,
        "forestPercent": round((forest_area / denominator) * 100.0, 4),
        "vegetationPercent": round((vegetation_area / denominator) * 100.0, 4),
        "nonVegetationPercent": round((non_vegetation_area / denominator) * 100.0, 4),
        "sparseVegetationPercent": round((sparse_vegetation_area / denominator) * 100.0, 4),
        "denseVegetationPercent": round((dense_vegetation_area / denominator) * 100.0, 4),
    }


def _build_maps(ndvi: ee.Image, classified: ee.Image, vegetation_mask: ee.Image, forest_mask: ee.Image) -> dict[str, str]:
    ndvi_tile_url = (
        ndvi.visualize(min=-0.2, max=0.8, palette=["#8c510a", "#f6e8c3", "#c7e9ad", "#1b7837"])
        .reproject(crs=WEB_MERCATOR, scale=30)
        .getMapId({})["tile_fetcher"]
        .url_format
    )
    classification_tile_url = (
        classified.visualize(min=0, max=2, palette=["#8c510a", "#f4d35e", "#1b7837"])
        .reproject(crs=WEB_MERCATOR, scale=30)
        .getMapId({})["tile_fetcher"]
        .url_format
    )
    vegetation_mask_tile_url = (
        vegetation_mask.selfMask()
        .visualize(min=0, max=1, palette=["#2e7d32"])
        .reproject(crs=WEB_MERCATOR, scale=30)
        .getMapId({})["tile_fetcher"]
        .url_format
    )
    forest_mask_tile_url = (
        forest_mask.selfMask()
        .visualize(min=0, max=1, palette=["#14532d"])
        .reproject(crs=WEB_MERCATOR, scale=30)
        .getMapId({})["tile_fetcher"]
        .url_format
    )

    return {
        "ndviTileUrl": ndvi_tile_url,
        "classificationTileUrl": classification_tile_url,
        "vegetationMaskTileUrl": vegetation_mask_tile_url,
        "forestMaskTileUrl": forest_mask_tile_url,
    }


def process(aoi: ee.Geometry | list[float] | tuple[float, float, float, float], year: int, month: int, day: int) -> dict[str, dict[str, object]]:
    init_ee()
    geometry = _normalize_aoi(aoi)

    composite, metadata = _get_composite(geometry, year, month, day)
    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("ndvi")
    classified, vegetation_mask = _classify_ndvi(ndvi)
    tree_mask = _get_tree_mask(geometry)
    forest_mask = tree_mask.And(ndvi.gt(VEGETATION_MASK_THRESHOLD)).updateMask(ndvi.mask()).rename("forestMask")

    stats = {
        **_compute_index_stats(composite, ndvi, geometry),
        **_compute_area_stats(ndvi, classified, vegetation_mask, forest_mask, geometry),
        **metadata,
    }
    maps = {
        **_build_maps(ndvi, classified, vegetation_mask, forest_mask),
        "source": str(metadata["source"]),
    }

    return {"stats": stats, "maps": maps}
