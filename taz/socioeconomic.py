"""Socioeconomic profile features and supervised TAZ classification.

The five supported labels describe built form and land use, following the
Gurugram project presentation.  They are not household-income estimates.
Feature extraction materializes one row per zone.  Large building rasters are
read in bounded windows and source files are never modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box, mapping
from shapely.ops import transform, unary_union
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeClassifier

from opengeomove.taz.exceptions import TAZDependencyError, TAZGeometryError
from opengeomove.taz.models import TAZFrame
from opengeomove.taz.population import attach_population


SOCIOECONOMIC_CLASSES: Mapping[int, str] = {
    0: "farmland_or_no_buildings",
    1: "urban_village",
    2: "open_areas_with_high_rise_buildings",
    3: "open_housing_mixed_with_rural_land",
    4: "rural_area_with_very_few_houses",
}
"""Stable label meanings used by the Gurugram supervised model."""


SOCIOECONOMIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "poi_orgs",
    "poi_offices",
    "poi_goods_store",
    "poi_health",
    "poi_food",
    "poi_living",
    "poi_education",
    "poi_total",
    "road_residential",
    "Area_Sq.Km",
    "Pop_sum",
    "PopDensity",
    "building_count_h_0_10_per_sqkm",
    "building_count_h_10_30_per_sqkm",
    "building_count_h_30_60_per_sqkm",
    "building_count_h_gt_60_per_sqkm",
    "height_weighted_mean",
    "count_times_height_sum_per_sqkm",
    "built_area_ratio",
    "empty_area_ratio",
    "height_p90",
    "height_p95",
    "height_p75",
    "height_p25",
    "building_count_sum",
)
"""Ordered model inputs compatible with the current Gurugram classifier."""


_POI_TYPES: Mapping[str, frozenset[str]] = {
    "orgs": frozenset({"association_or_organization"}),
    "offices": frozenset(
        {
            "corporate_office", "business_center", "coworking_space", "lawyer",
            "roofing_contractor", "tour_agency", "shipping_service",
            "telecommunications_service_provider", "television_studio",
        }
    ),
    "goods_store": frozenset(
        {
            "discount_store", "sporting_goods_store", "sports_bar",
            "department_store", "general_store", "grocery_store",
            "convenience_store", "hypermarket", "farmers_market",
            "home_goods_store", "building_materials_store", "asian_grocery_store",
        }
    ),
    "health": frozenset(
        {
            "pharmacy", "dentist", "hospital", "clinic", "doctors", "blood_bank",
            "doctor", "drugstore", "medical_clinic", "pet_store",
            "pet_boarding_service", "dental_clinic", "chiropractor",
            "medical_center", "health", "pet_care", "skin_care_clinic",
            "medical_lab", "foot_care", "veterinary_care", "general_hospital",
        }
    ),
    "food": frozenset(
        {
            "south_indian_restaurant", "restaurant", "cafe", "fast_food",
            "food_court", "ice_cream", "internet_cafe", "canteen", "bbq",
            "ramen_restaurant", "chinese_noodle_restaurant", "meal_takeaway",
            "food_delivery", "pizza_delivery", "ice_cream_shop",
            "middle_eastern_restaurant", "gyro_restaurant", "donut_shop",
            "burmese_restaurant", "chocolate_shop", "dessert_restaurant",
            "asian_restaurant", "fine_dining_restaurant", "seafood_restaurant",
            "vegetarian_restaurant", "french_restaurant", "coffee_stand",
            "korean_restaurant", "buffet_restaurant", "italian_restaurant",
            "tex_mex_restaurant", "sushi_restaurant", "butcher_shop",
            "breakfast_restaurant", "mexican_restaurant", "cocktail_bar",
            "fast_food_restaurant", "dumpling_restaurant", "chinese_restaurant",
            "coffee_shop", "pizza_restaurant", "bed_and_breakfast", "pastry_shop",
            "chicken_restaurant", "soul_food_restaurant",
            "south_american_restaurant", "deli", "vegan_restaurant",
            "american_restaurant", "german_restaurant", "mediterranean_restaurant",
            "barbecue_restaurant", "thai_restaurant", "food",
            "afghani_restaurant", "indian_restaurant", "tea_house", "bistro",
            "tea_store", "tibetan_restaurant", "western_restaurant", "brewpub",
            "snack_bar", "health_food_store", "bakery", "catering_service",
            "hot_dog_stand", "shawarma_restaurant", "bar_and_grill",
            "barbecue_area", "family_restaurant", "kebab_shop", "cafeteria",
            "coffee_roastery", "juice_shop", "indonesian_restaurant",
            "japanese_restaurant", "north_indian_restaurant", "european_restaurant",
            "food_store", "salad_shop", "dog_cafe", "diner",
            "hamburger_restaurant", "steak_house", "brewery", "meal_delivery",
            "sandwich_shop", "southwestern_us_restaurant", "brunch_restaurant",
            "fusion_restaurant", "lebanese_restaurant", "candy_store", "cake_shop",
            "dessert_shop", "confectionery", "pub", "bar", "night_club",
            "hookah_bar", "gastropub", "lounge_bar", "liquor_store", "wine_bar",
        }
    ),
    "living": frozenset(
        {
            "lodging", "extended_stay_hotel", "japanese_inn", "apartment_complex",
            "apartment_building", "condominium_complex", "hotel", "camping_cabin",
            "motel", "cottage", "hostel", "resort_hotel", "inn",
            "housing_complex", "guest_house", "private_guest_room",
        }
    ),
    "education": frozenset(
        {
            "school", "kindergarten", "university", "college", "library",
            "research_institute", "prep_school", "academic_department",
            "educational_institution", "secondary_school", "primary_school",
        }
    ),
}


def _metric_zones(zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Project zones to a locally estimated metric CRS."""
    units = {
        (axis.unit_name or "").lower()
        for axis in getattr(zones.crs, "axis_info", ())
    }
    if (
        not getattr(zones.crs, "is_geographic", False)
        and units
        and units <= {"metre", "meter"}
    ):
        return zones.copy()
    metric_crs = zones.estimate_utm_crs()
    if metric_crs is None:
        raise TAZGeometryError("Could not estimate a metric CRS for socioeconomic features")
    return zones.to_crs(metric_crs)


def _normalize_zones(zones: TAZFrame | gpd.GeoDataFrame, id_col: str) -> TAZFrame:
    """Return arbitrary polygon zones as a validated TAZFrame."""
    if isinstance(zones, TAZFrame):
        data = zones.to_geopandas()
        metadata = zones.metadata
    elif isinstance(zones, gpd.GeoDataFrame):
        data = zones.copy()
        metadata = {}
    else:
        raise TypeError("zones must be a TAZFrame or GeoDataFrame")
    if data.crs is None:
        raise TAZGeometryError("Socioeconomic zones require an explicit CRS")
    if id_col not in data:
        raise ValueError(f"Zone ID column {id_col!r} is missing")
    if data[id_col].isna().any() or data[id_col].astype(str).duplicated().any():
        raise ValueError(f"Zone ID column {id_col!r} must be non-null and unique")
    data["taz_id"] = data[id_col].astype(str)
    metric = _metric_zones(data)
    data["area_km2"] = metric.geometry.area.to_numpy(dtype=float) / 1_000_000
    return TAZFrame(data, {**metadata, "source_id_column": id_col})


def _validate_spatial_source(source: gpd.GeoDataFrame, name: str) -> gpd.GeoDataFrame:
    """Copy a non-empty spatial source after validating its CRS."""
    if not isinstance(source, gpd.GeoDataFrame):
        raise TypeError(f"{name} must be a GeoDataFrame")
    if source.crs is None:
        raise TAZGeometryError(f"{name} requires an explicit CRS")
    return source[source.geometry.notna() & ~source.geometry.is_empty].copy()


def _add_poi_features(
    result: gpd.GeoDataFrame,
    poi: gpd.GeoDataFrame,
    *,
    poi_type_col: str,
) -> None:
    """Add legacy-compatible point-of-interest counts in place."""
    points = _validate_spatial_source(poi, "poi").to_crs(result.crs)
    if poi_type_col not in points:
        raise ValueError(f"POI type column {poi_type_col!r} is missing")
    joined = gpd.sjoin(
        points[[poi_type_col, "geometry"]],
        result[["taz_id", "geometry"]],
        how="inner",
        predicate="within",
    )
    totals = joined.groupby("taz_id", sort=False).size()
    result["poi_total"] = result["taz_id"].map(totals).fillna(0).astype(int)
    normalized = joined[poi_type_col].astype("string").str.strip().str.lower()
    for category, accepted in _POI_TYPES.items():
        counts = joined.loc[normalized.isin(accepted)].groupby("taz_id", sort=False).size()
        result[f"poi_{category}"] = result["taz_id"].map(counts).fillna(0).astype(int)


def _highway_tokens(value: Any) -> frozenset[str]:
    """Normalize scalar, sequence, or serialized OSM highway values."""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return frozenset(str(item).strip().lower() for item in value)
    text = str(value).strip().lower().strip("[]")
    return frozenset(part.strip(" '\"") for part in text.split(",") if part.strip(" '\""))


def _is_residential_road(value: Any) -> bool:
    """Apply the road grouping used to train the Gurugram model."""
    tokens = _highway_tokens(value)
    if "residential" in tokens or (len(tokens) == 1 and tokens & {"path", "corridor", "pedestrian", "steps"}):
        return True
    return tokens in {
        frozenset({"corridor", "footway"}),
        frozenset({"path", "steps", "footway"}),
        frozenset({"steps", "corridor", "footway"}),
        frozenset({"path", "service"}),
        frozenset({"steps", "pedestrian"}),
        frozenset({"path", "steps"}),
    }


def _add_road_features(
    result: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    *,
    road_type_col: str,
) -> None:
    """Add intersecting residential-road record counts in place."""
    edges = _validate_spatial_source(roads, "roads").to_crs(result.crs)
    if road_type_col not in edges:
        raise ValueError(f"Road type column {road_type_col!r} is missing")
    residential = edges[edges[road_type_col].apply(_is_residential_road)]
    joined = gpd.sjoin(
        residential[["geometry"]],
        result[["taz_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    counts = joined.groupby("taz_id", sort=False).size()
    result["road_residential"] = result["taz_id"].map(counts).fillna(0).astype(int)


def _add_population_features(
    zones: TAZFrame,
    result: gpd.GeoDataFrame,
    population: Any,
    *,
    population_column: str,
    population_allocation: str,
) -> None:
    """Add population totals and densities in place."""
    if population is not None:
        populated = attach_population(
            zones,
            population,
            output_column="Pop_sum",
            allocation=population_allocation,
            population_column=population_column,
        )
        values = populated.data.set_index("taz_id")["Pop_sum"]
        result["Pop_sum"] = result["taz_id"].map(values).fillna(0.0)
    elif population_column in result:
        result["Pop_sum"] = pd.to_numeric(result[population_column], errors="coerce")
    elif "Pop_sum" in result:
        result["Pop_sum"] = pd.to_numeric(result["Pop_sum"], errors="coerce")
    else:
        raise ValueError(
            "population is required unless zones contain population_column or 'Pop_sum'"
        )
    population_values = result["Pop_sum"].to_numpy(dtype=float)
    if not np.isfinite(population_values).all() or (population_values < 0).any():
        raise ValueError("Population values must be finite and non-negative")
    result["PopDensity"] = result["Pop_sum"] / result["Area_Sq.Km"]


def _add_footprint_features(
    result: gpd.GeoDataFrame,
    footprints: gpd.GeoDataFrame,
    *,
    confidence_col: str | None,
    confidence_threshold: float,
) -> None:
    """Add building-footprint and empty-area ratios in place."""
    buildings = _validate_spatial_source(footprints, "building_footprints")
    if confidence_col is not None:
        if confidence_col not in buildings:
            raise ValueError(f"Building confidence column {confidence_col!r} is missing")
        confidence = pd.to_numeric(buildings[confidence_col], errors="coerce")
        buildings = buildings[confidence > confidence_threshold].copy()
    metric_zones = _metric_zones(result)
    buildings = buildings.to_crs(metric_zones.crs)
    if buildings.empty:
        area_by_zone = pd.Series(dtype=float)
    else:
        buildings["_footprint_area"] = buildings.geometry.area
        buildings.geometry = buildings.geometry.representative_point()
        joined = gpd.sjoin(
            buildings[["_footprint_area", "geometry"]],
            metric_zones[["taz_id", "geometry"]],
            how="inner",
            predicate="within",
        )
        area_by_zone = joined.groupby("taz_id", sort=False)["_footprint_area"].sum()
    zone_area = result["Area_Sq.Km"].to_numpy(dtype=float) * 1_000_000
    built_area = result["taz_id"].map(area_by_zone).fillna(0.0).to_numpy(dtype=float)
    built_area = np.minimum(built_area, zone_area)
    result["built_area_ratio"] = built_area / zone_area
    result["empty_area_ratio"] = np.maximum(0.0, zone_area - built_area) / zone_area


def _raster_paths(source: str | Path | Sequence[str | Path]) -> list[Path]:
    """Resolve one raster, a raster directory, or a path sequence."""
    raw: Iterable[str | Path]
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if path.is_dir():
            raw = sorted((*path.glob("*.tif"), *path.glob("*.tiff")))
        else:
            raw = [path]
    else:
        raw = source
    paths = sorted({Path(item).expanduser().resolve() for item in raw})
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Building raster files do not exist: {missing}")
    if not paths:
        raise ValueError("building_rasters did not resolve to any TIFF files")
    return paths


def _histogram_percentile(histogram: np.ndarray, percentile: float) -> float:
    """Estimate a height percentile from quarter-metre bounded bins."""
    total = int(histogram.sum())
    if total == 0:
        return float("nan")
    rank = max(1, int(np.ceil(percentile / 100 * total)))
    index = int(np.searchsorted(np.cumsum(histogram), rank, side="left"))
    return min(index, len(histogram) - 1) * 0.25


def _add_raster_features(
    result: gpd.GeoDataFrame,
    raster_source: str | Path | Sequence[str | Path],
    *,
    presence_threshold: float,
) -> None:
    """Aggregate Google 2.5D building bands with bounded window reads."""
    try:
        import rasterio
        from pyproj import Transformer
        from rasterio.features import rasterize
        from rasterio.windows import Window, bounds as window_bounds, from_bounds
    except ImportError as exc:
        raise TAZDependencyError(
            "Building-raster features require rasterio and pyproj"
        ) from exc

    count_sum = np.zeros(len(result), dtype=float)
    weighted_sum = np.zeros(len(result), dtype=float)
    height_bands = np.zeros((len(result), 4), dtype=float)
    histograms: dict[int, np.ndarray] = {}
    used_wgs84 = None

    for path in _raster_paths(raster_source):
        with rasterio.open(path) as dataset:
            if dataset.count < 3:
                raise ValueError(f"Building raster {path} requires count, height, and presence bands")
            if dataset.crs is None:
                raise TAZGeometryError(f"Building raster {path} requires a CRS")
            zones = result[["geometry"]].to_crs(dataset.crs).reset_index(drop=True)
            raster_footprint = box(*dataset.bounds)
            to_wgs84 = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
            from_wgs84 = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
            footprint_wgs84 = transform(to_wgs84.transform, raster_footprint)
            available_wgs84 = footprint_wgs84 if used_wgs84 is None else footprint_wgs84.difference(used_wgs84)
            if available_wgs84.is_empty:
                continue
            available = transform(from_wgs84.transform, available_wgs84)
            positions = zones.sindex.query(available, predicate="intersects")
            if len(positions) == 0:
                used_wgs84 = footprint_wgs84 if used_wgs84 is None else unary_union([used_wgs84, footprint_wgs84])
                continue
            requested = from_bounds(*available.bounds, transform=dataset.transform).round_offsets().round_lengths()
            try:
                crop = requested.intersection(Window(0, 0, dataset.width, dataset.height))
            except rasterio.errors.WindowError:
                continue
            source_block_height, source_block_width = dataset.block_shapes[0]
            block_height = min(1024, max(512, source_block_height))
            block_width = min(1024, max(512, source_block_width))
            row_end = int(crop.row_off + crop.height)
            col_end = int(crop.col_off + crop.width)
            for row_off in range(int(crop.row_off), row_end, block_height):
                for col_off in range(int(crop.col_off), col_end, block_width):
                    window = Window(
                        col_off,
                        row_off,
                        min(block_width, col_end - col_off),
                        min(block_height, row_end - row_off),
                    )
                    window_geometry = box(*window_bounds(window, dataset.transform))
                    local_positions = zones.sindex.query(window_geometry, predicate="intersects")
                    if len(local_positions) == 0 or not available.intersects(window_geometry):
                        continue
                    data = dataset.read([1, 2, 3], window=window, masked=True)
                    window_transform = dataset.window_transform(window)
                    zone_grid = rasterize(
                        [(zones.geometry.iloc[int(pos)], int(pos) + 1) for pos in local_positions],
                        out_shape=data.shape[1:],
                        transform=window_transform,
                        fill=0,
                        dtype="int32",
                    )
                    available_grid = rasterize(
                        [(mapping(available), 1)],
                        out_shape=data.shape[1:],
                        transform=window_transform,
                        fill=0,
                        dtype="uint8",
                    )
                    masks = np.ma.getmaskarray(data)
                    values = np.asarray(data.filled(np.nan), dtype=float)
                    valid = (
                        ~masks.any(axis=0)
                        & np.isfinite(values).all(axis=0)
                        & (values[2] >= presence_threshold)
                        & (zone_grid > 0)
                        & (available_grid > 0)
                    )
                    zone_values = zone_grid[valid] - 1
                    counts = values[0][valid]
                    heights = values[1][valid]
                    for position in np.unique(zone_values):
                        keep = zone_values == position
                        zone_counts = counts[keep]
                        zone_heights = heights[keep]
                        position = int(position)
                        count_sum[position] += float(zone_counts.sum())
                        weighted_sum[position] += float((zone_counts * zone_heights).sum())
                        height_bands[position, 0] += float(zone_counts[(zone_heights >= 0) & (zone_heights < 10)].sum())
                        height_bands[position, 1] += float(zone_counts[(zone_heights >= 10) & (zone_heights < 30)].sum())
                        height_bands[position, 2] += float(zone_counts[(zone_heights >= 30) & (zone_heights <= 60)].sum())
                        height_bands[position, 3] += float(zone_counts[zone_heights > 60].sum())
                        histogram = histograms.setdefault(position, np.zeros(4001, dtype=np.int64))
                        indices = np.clip(np.rint(zone_heights / 0.25).astype(int), 0, 4000)
                        histogram += np.bincount(indices, minlength=4001)
            used_wgs84 = footprint_wgs84 if used_wgs84 is None else unary_union([used_wgs84, footprint_wgs84])

    area = result["Area_Sq.Km"].to_numpy(dtype=float)
    result["building_count_sum"] = count_sum
    result["building_count_h_0_10_per_sqkm"] = height_bands[:, 0] / area
    result["building_count_h_10_30_per_sqkm"] = height_bands[:, 1] / area
    result["building_count_h_30_60_per_sqkm"] = height_bands[:, 2] / area
    result["building_count_h_gt_60_per_sqkm"] = height_bands[:, 3] / area
    result["height_weighted_mean"] = np.divide(
        weighted_sum,
        count_sum,
        out=np.full(len(result), np.nan),
        where=count_sum > 0,
    )
    result["count_times_height_sum_per_sqkm"] = weighted_sum / area
    for percentile in (25, 75, 90, 95):
        result[f"height_p{percentile}"] = [
            _histogram_percentile(histograms.get(index, np.zeros(1, dtype=int)), percentile)
            for index in range(len(result))
        ]


def extract_socioeconomic_features(
    zones: TAZFrame | gpd.GeoDataFrame,
    *,
    poi: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    population: Any = None,
    building_footprints: gpd.GeoDataFrame,
    building_rasters: str | Path | Sequence[str | Path],
    id_col: str = "taz_id",
    poi_type_col: str = "primaryType",
    road_type_col: str = "highway",
    population_column: str = "population",
    population_allocation: str = "area_weighted",
    footprint_confidence_col: str | None = "confidence",
    footprint_confidence_threshold: float = 0.5,
    building_presence_threshold: float = 0.5,
) -> TAZFrame:
    """Extract the 25 features used by the socioeconomic decision tree.

    Args:
        zones: Arbitrary non-overlapping polygon zones as a TAZFrame or
            GeoDataFrame.
        poi: Point GeoDataFrame with Google-style POI type values.
        roads: Line GeoDataFrame with OSM-style highway values.
        population: Optional population cells or raster accepted by
            :func:`attach_population`. If omitted, population is read from the
            zone ``population_column`` or ``Pop_sum`` field.
        building_footprints: Polygon building footprints used for built-area
            and empty-area ratios.
        building_rasters: Google Open Buildings 2.5D TIFF path, directory, or
            sequence. Bands 1, 2, and 3 must be count, height, and presence.
        id_col: Unique identifier in ``zones``.
        poi_type_col: POI category field.
        road_type_col: OSM highway field.
        population_column: Population field on zones or vector cells.
        population_allocation: ``area_weighted`` or ``cell_centroid``.
        footprint_confidence_col: Optional footprint-confidence field. Set to
            ``None`` when the source has no confidence values.
        footprint_confidence_threshold: Strict minimum footprint confidence.
        building_presence_threshold: Inclusive minimum presence probability.

    Returns:
        A newly materialized TAZFrame containing geometry, source fields, and
        all columns in :data:`SOCIOECONOMIC_FEATURE_COLUMNS`.

    Raises:
        TypeError: If a spatial input has the wrong object type.
        ValueError: If IDs or required source fields are missing.
        TAZGeometryError: If a spatial input lacks a CRS.

    Notes:
        Raster height percentiles use deterministic 0.25-metre histogram bins
        bounded at 1,000 metres. This keeps memory bounded for city-scale data.
        Overlapping raster tiles are de-duplicated in input path order.
    """
    if not 0 <= footprint_confidence_threshold <= 1:
        raise ValueError("footprint_confidence_threshold must be between 0 and 1")
    if not 0 <= building_presence_threshold <= 1:
        raise ValueError("building_presence_threshold must be between 0 and 1")
    normalized = _normalize_zones(zones, id_col)
    result = normalized.to_geopandas()
    result["Area_Sq.Km"] = result["area_km2"]
    _add_population_features(
        normalized,
        result,
        population,
        population_column=population_column,
        population_allocation=population_allocation,
    )
    _add_poi_features(result, poi, poi_type_col=poi_type_col)
    _add_road_features(result, roads, road_type_col=road_type_col)
    _add_footprint_features(
        result,
        building_footprints,
        confidence_col=footprint_confidence_col,
        confidence_threshold=footprint_confidence_threshold,
    )
    _add_raster_features(
        result,
        building_rasters,
        presence_threshold=building_presence_threshold,
    )
    return TAZFrame(
        result,
        {
            **normalized.metadata,
            "socioeconomic_feature_columns": list(SOCIOECONOMIC_FEATURE_COLUMNS),
            "socioeconomic_feature_rows": len(result),
            "building_presence_threshold": building_presence_threshold,
            "footprint_confidence_threshold": footprint_confidence_threshold,
        },
    )


def _table(value: TAZFrame | pd.DataFrame) -> pd.DataFrame:
    """Return a copied pandas-compatible table without geometry assumptions."""
    if isinstance(value, TAZFrame):
        return value.to_geopandas()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    raise TypeError("features must be a TAZFrame, GeoDataFrame, or DataFrame")


def _annotation_id(annotations: pd.DataFrame, requested: str) -> str:
    """Resolve the annotation ID while supporting the legacy lowercase name."""
    if requested in annotations:
        return requested
    if "taz_id" in annotations:
        return "taz_id"
    matches = [column for column in annotations if column.lower() == requested.lower()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Annotations do not contain ID column {requested!r} or 'taz_id'")


class SocioeconomicClassifier:
    """A fitted, uncalibrated decision-tree profile classifier.

    Instances are created by :func:`fit_socioeconomic_classifier`. Predictions
    retain the input table type; TAZFrame predictions also retain metadata.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        feature_columns: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> None:
        """Store a fitted pipeline, ordered features, and training metadata."""
        self.pipeline = pipeline
        self.feature_columns = tuple(feature_columns)
        self.metadata = dict(metadata)

    @property
    def classes_(self) -> np.ndarray:
        """Return the integer class labels learned by the decision tree."""
        return self.pipeline.named_steps["decisiontreeclassifier"].classes_.copy()

    def predict(self, features: TAZFrame | pd.DataFrame) -> TAZFrame | pd.DataFrame:
        """Predict profiles and append labels, names, and raw probabilities.

        Args:
            features: TAZFrame, GeoDataFrame, or DataFrame containing every
                ordered model feature.

        Returns:
            A copied object of the same type. The result contains
            ``socioeconomic_label``, ``socioeconomic_class``, and one raw
            probability column per class. Decision-tree probabilities are not
            calibrated confidence estimates.

        Raises:
            ValueError: If a model feature is missing or non-numeric.
        """
        table = _table(features)
        missing = [column for column in self.feature_columns if column not in table]
        if missing:
            raise ValueError(f"Missing socioeconomic feature columns: {missing}")
        matrix = table.loc[:, self.feature_columns].apply(pd.to_numeric, errors="coerce")
        matrix = matrix.replace([np.inf, -np.inf], np.nan)
        labels = self.pipeline.predict(matrix).astype(int)
        probabilities = self.pipeline.predict_proba(matrix)
        table["socioeconomic_label"] = labels
        table["socioeconomic_class"] = [SOCIOECONOMIC_CLASSES[int(label)] for label in labels]
        for label, probability in zip(self.classes_, probabilities.T):
            table[f"socioeconomic_probability_{int(label)}"] = probability
        if isinstance(features, TAZFrame):
            return TAZFrame(
                table,
                {
                    **features.metadata,
                    "socioeconomic_classification": self.metadata,
                },
            )
        if isinstance(features, gpd.GeoDataFrame):
            return gpd.GeoDataFrame(table, geometry=features.geometry.name, crs=features.crs)
        return table


def fit_socioeconomic_classifier(
    features: TAZFrame | pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    id_col: str = "taz_id",
    label_col: str = "label",
    random_state: int = 42,
    evaluation_folds: int | None = 5,
) -> SocioeconomicClassifier:
    """Fit the approved 25-feature Gurugram decision-tree workflow.

    Args:
        features: TAZFrame, GeoDataFrame, or DataFrame with unique IDs and all
            socioeconomic feature columns.
        annotations: In-memory ground-truth table. Its ID may use ``id_col`` or
            the legacy ``taz_id`` spelling.
        id_col: Identifier field in ``features``.
        label_col: Integer annotation field using labels 0 through 4.
        random_state: Deterministic decision-tree and fold seed.
        evaluation_folds: Optional stratified cross-validation fold count.
            Set to ``None`` to skip evaluation. Evaluation materializes only
            matched labelled feature rows.

    Returns:
        A fitted SocioeconomicClassifier. It uses median imputation and a
        class-balanced, otherwise unpruned DecisionTreeClassifier to preserve
        the existing model contract.

    Raises:
        TypeError: If annotations is not a DataFrame.
        ValueError: If IDs, labels, or selected features are invalid.

    Notes:
        ``urban_village_gt`` is deliberately excluded from model inputs to
        prevent ground-truth leakage. Cross-validation scores are diagnostic;
        random folds do not measure geographic transfer to another city.
    """
    table = _table(features)
    if not isinstance(annotations, pd.DataFrame):
        raise TypeError("annotations must be a DataFrame")
    if id_col not in table:
        raise ValueError(f"Feature ID column {id_col!r} is missing")
    missing = [column for column in SOCIOECONOMIC_FEATURE_COLUMNS if column not in table]
    if missing:
        raise ValueError(f"Missing socioeconomic feature columns: {missing}")
    if table[id_col].isna().any() or table[id_col].astype(str).duplicated().any():
        raise ValueError(f"Feature ID column {id_col!r} must be non-null and unique")
    annotation_id = _annotation_id(annotations, id_col)
    if label_col not in annotations:
        raise ValueError(f"Annotation label column {label_col!r} is missing")
    labels = annotations[[annotation_id, label_col]].copy()
    labels = labels.drop_duplicates(annotation_id, keep="last")
    labels["_join_id"] = labels[annotation_id].astype(str)
    labels[label_col] = pd.to_numeric(labels[label_col], errors="raise")
    if labels[label_col].isna().any() or not np.equal(labels[label_col] % 1, 0).all():
        raise ValueError("Socioeconomic labels must be integers")
    labels[label_col] = labels[label_col].astype(int)
    unknown = sorted(set(labels[label_col]) - set(SOCIOECONOMIC_CLASSES))
    if unknown:
        raise ValueError(f"Unknown socioeconomic labels: {unknown}")
    prepared = table.copy()
    prepared["_join_id"] = prepared[id_col].astype(str)
    labelled = prepared.merge(labels[["_join_id", label_col]], on="_join_id", how="inner")
    if labelled.empty:
        raise ValueError("No annotation IDs matched the feature table")
    if labelled[label_col].nunique() < 2:
        raise ValueError("At least two annotated classes are required")
    matrix = labelled.loc[:, SOCIOECONOMIC_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    target = labelled[label_col].to_numpy(dtype=int)
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
    )
    pipeline.fit(matrix, target)
    training_predictions = pipeline.predict(matrix)
    metadata: dict[str, Any] = {
        "feature_columns": list(SOCIOECONOMIC_FEATURE_COLUMNS),
        "class_definitions": dict(SOCIOECONOMIC_CLASSES),
        "annotation_rows": int(len(labelled)),
        "class_counts": {int(key): int(value) for key, value in labelled[label_col].value_counts().sort_index().items()},
        "training_accuracy": float(accuracy_score(target, training_predictions)),
        "training_macro_f1": float(f1_score(target, training_predictions, average="macro")),
        "random_state": random_state,
        "probabilities_calibrated": False,
    }
    if evaluation_folds is not None:
        if evaluation_folds < 2:
            raise ValueError("evaluation_folds must be at least 2 or None")
        effective_folds = min(
            evaluation_folds,
            int(pd.Series(target).value_counts().min()),
        )
        if effective_folds >= 2:
            split = StratifiedKFold(
                n_splits=effective_folds,
                shuffle=True,
                random_state=random_state,
            )
            scores = cross_validate(
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
                ),
                matrix,
                target,
                cv=split,
                scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
            )
            metadata["cross_validation"] = {
                "requested_folds": evaluation_folds,
                "folds": effective_folds,
                "mean_accuracy": float(scores["test_accuracy"].mean()),
                "mean_macro_f1": float(scores["test_macro_f1"].mean()),
            }
        else:
            metadata["cross_validation"] = {
                "requested_folds": evaluation_folds,
                "folds": 0,
                "reason": "At least two rows per annotated class are required",
            }
    return SocioeconomicClassifier(
        pipeline,
        feature_columns=SOCIOECONOMIC_FEATURE_COLUMNS,
        metadata=metadata,
    )


__all__ = [
    "SOCIOECONOMIC_CLASSES",
    "SOCIOECONOMIC_FEATURE_COLUMNS",
    "SocioeconomicClassifier",
    "extract_socioeconomic_features",
    "fit_socioeconomic_classifier",
]
