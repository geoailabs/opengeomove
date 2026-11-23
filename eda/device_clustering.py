"""Device-level mobility profiles and reproducible KMeans clustering.

The device profile is designed for the trajectory-visit table produced by the
Gurugram mobility workflow. Feature construction stays in DuckDB; only the
small device-by-feature matrix required by scikit-learn is materialized for
KMeans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

if TYPE_CHECKING:
    import duckdb

    from opengeomove.data_loading.dataset import Dataset


DEVICE_CLUSTER_FEATURES: tuple[str, ...] = (
    "n_active_days",
    "median_daily_distance_km",
    "mean_daily_distance_km",
    "p90_daily_distance_km",
    "sd_daily_distance_km",
    "mean_stay_points_per_trajectory",
    "min_stay_points_per_trajectory",
    "max_stay_points_per_trajectory",
    "p90_stay_points_per_trajectory",
    "share_trip_100m_2km",
    "share_trip_2_5km",
    "share_trip_5_10km",
    "share_trip_10_30km",
    "share_trip_30km_plus",
    "number_of_grids_visited",
    "anchor_grid_distance_min_km",
    "anchor_grid_distance_p50_km",
    "anchor_grid_distance_p75_km",
    "anchor_grid_distance_p90_km",
    "anchor_grid_distance_max_km",
)

DEVICE_CLUSTER_LABELS: Mapping[int, str] = {
    0: "Moderate multi-purpose local commuters",
    1: "Low-mobility neighbourhood walkers",
    2: "Occasional mid-distance outward commuters",
    3: "Highly active regional commuters",
    4: "Highly active local urban movers",
}

_OUTPUT_IDS = itertools.count()
_EARTH_RADIUS_KM = 6_371.0088


def _quote_identifier(name: str) -> str:
    """Return a safely quoted SQL identifier."""
    if not isinstance(name, str) or not name:
        raise TypeError("column names must be non-empty strings")
    return '"' + name.replace('"', '""') + '"'


def _require_columns(dataset: Dataset, columns: Sequence[str]) -> None:
    """Raise a useful error when a requested source column is absent."""
    missing = [column for column in columns if column not in dataset.columns]
    if missing:
        raise ValueError(f"Columns do not exist in this Dataset: {missing!r}")


def _validate_dataset(dataset: Dataset) -> None:
    """Validate the common Dataset contract without importing it at module load."""
    from opengeomove.data_loading.dataset import Dataset as DatasetType

    if not isinstance(dataset, DatasetType):
        raise TypeError("dataset must be an opengeomove.Dataset")


def _trajectory_feature_relation(
    dataset: Dataset,
    *,
    device_id_col: str,
    trajectory_id_col: str,
    trajectory_date_col: str,
    trajectory_distance_col: str,
    visit_order_col: str,
    stay_point_col: str,
    grid_col: str,
    lat_col: str,
    lon_col: str,
    trajectory_start_col: str,
) -> duckdb.DuckDBPyRelation:
    """Build the DuckDB relation containing one row per device profile."""
    quoted = {
        "device": _quote_identifier(device_id_col),
        "trajectory": _quote_identifier(trajectory_id_col),
        "date": _quote_identifier(trajectory_date_col),
        "distance": _quote_identifier(trajectory_distance_col),
        "visit_order": _quote_identifier(visit_order_col),
        "stay_point": _quote_identifier(stay_point_col),
        "grid": _quote_identifier(grid_col),
        "lat": _quote_identifier(lat_col),
        "lon": _quote_identifier(lon_col),
        "start": _quote_identifier(trajectory_start_col),
    }
    source_alias = f"_ogm_device_profile_source_{next(_OUTPUT_IDS)}"
    query = f"""
        WITH raw AS (
            SELECT
                CAST({quoted['device']} AS VARCHAR) AS device_key,
                CAST({quoted['trajectory']} AS VARCHAR) AS trajectory_key,
                TRY_CAST({quoted['date']} AS DATE) AS trajectory_date,
                TRY_CAST({quoted['distance']} AS DOUBLE) AS distance_m,
                TRY_CAST({quoted['visit_order']} AS BIGINT) AS visit_order,
                TRY_CAST({quoted['stay_point']} AS BIGINT) AS stay_point,
                CAST({quoted['grid']} AS VARCHAR) AS grid_key,
                TRY_CAST({quoted['lat']} AS DOUBLE) AS latitude,
                TRY_CAST({quoted['lon']} AS DOUBLE) AS longitude,
                TRY_CAST({quoted['start']} AS TIMESTAMP) AS trajectory_start
            FROM {source_alias}
        ),
        base AS (
            SELECT *,
                COALESCE(
                    trajectory_start,
                    CAST(trajectory_date AS TIMESTAMP)
                ) AS anchor_order
            FROM raw
            WHERE device_key IS NOT NULL
              AND trajectory_key IS NOT NULL
              AND trajectory_date IS NOT NULL
        ),
        trajectories AS (
            SELECT
                device_key,
                trajectory_key,
                MIN(trajectory_date) AS trajectory_date,
                MAX(distance_m) FILTER (
                    WHERE distance_m IS NOT NULL AND isfinite(distance_m)
                ) AS distance_m,
                SUM(CASE WHEN stay_point = 1 THEN 1 ELSE 0 END)::DOUBLE
                    AS stay_points
            FROM base
            GROUP BY device_key, trajectory_key
        ),
        daily AS (
            SELECT
                device_key,
                trajectory_date,
                SUM(COALESCE(distance_m, 0.0)) / 1000.0
                    AS daily_distance_km
            FROM trajectories
            GROUP BY device_key, trajectory_date
        ),
        daily_profile AS (
            SELECT
                device_key,
                COUNT(*)::BIGINT AS n_active_days,
                QUANTILE_CONT(daily_distance_km, 0.5) AS median_daily_distance_km,
                AVG(daily_distance_km) AS mean_daily_distance_km,
                QUANTILE_CONT(daily_distance_km, 0.9) AS p90_daily_distance_km,
                COALESCE(STDDEV_POP(daily_distance_km), 0.0)
                    AS sd_daily_distance_km
            FROM daily
            GROUP BY device_key
        ),
        trajectory_profile AS (
            SELECT
                device_key,
                AVG(stay_points) AS mean_stay_points_per_trajectory,
                MIN(stay_points) AS min_stay_points_per_trajectory,
                MAX(stay_points) AS max_stay_points_per_trajectory,
                QUANTILE_CONT(stay_points, 0.9)
                    AS p90_stay_points_per_trajectory,
                COUNT(*) FILTER (
                    WHERE distance_m IS NOT NULL AND distance_m >= 100
                )::DOUBLE AS qualifying_trips
            FROM trajectories
            GROUP BY device_key
        ),
        trip_shares AS (
            SELECT
                device_key,
                COUNT(*) FILTER (
                    WHERE distance_m >= 100 AND distance_m < 2000
                ) / NULLIF(qualifying_trips, 0.0)
                    AS share_trip_100m_2km,
                COUNT(*) FILTER (
                    WHERE distance_m >= 2000 AND distance_m < 5000
                ) / NULLIF(qualifying_trips, 0.0)
                    AS share_trip_2_5km,
                COUNT(*) FILTER (
                    WHERE distance_m >= 5000 AND distance_m < 10000
                ) / NULLIF(qualifying_trips, 0.0)
                    AS share_trip_5_10km,
                COUNT(*) FILTER (
                    WHERE distance_m >= 10000 AND distance_m < 30000
                ) / NULLIF(qualifying_trips, 0.0)
                    AS share_trip_10_30km,
                COUNT(*) FILTER (
                    WHERE distance_m >= 30000
                ) / NULLIF(qualifying_trips, 0.0)
                    AS share_trip_30km_plus
            FROM trajectories
            JOIN trajectory_profile USING (device_key)
            GROUP BY device_key, qualifying_trips
        ),
        grid_centroids AS (
            SELECT
                device_key,
                grid_key,
                AVG(latitude) AS grid_latitude,
                AVG(longitude) AS grid_longitude
            FROM base
            WHERE grid_key IS NOT NULL
              AND latitude IS NOT NULL AND isfinite(latitude)
              AND longitude IS NOT NULL AND isfinite(longitude)
              AND latitude BETWEEN -90 AND 90
              AND longitude BETWEEN -180 AND 180
            GROUP BY device_key, grid_key
        ),
        anchor_candidates AS (
            SELECT
                device_key,
                latitude AS anchor_latitude,
                longitude AS anchor_longitude,
                ROW_NUMBER() OVER (
                    PARTITION BY device_key
                    ORDER BY anchor_order NULLS LAST,
                             trajectory_key,
                             visit_order NULLS LAST,
                             latitude NULLS LAST,
                             longitude NULLS LAST
                ) AS row_number
            FROM base
            WHERE latitude IS NOT NULL AND isfinite(latitude)
              AND longitude IS NOT NULL AND isfinite(longitude)
              AND latitude BETWEEN -90 AND 90
              AND longitude BETWEEN -180 AND 180
        ),
        anchors AS (
            SELECT * EXCLUDE (row_number)
            FROM anchor_candidates
            WHERE row_number = 1
        ),
        anchor_distances AS (
            SELECT
                grids.device_key,
                grids.grid_key,
                2.0 * {_EARTH_RADIUS_KM} * ASIN(SQRT(LEAST(1.0,
                    POWER(SIN(RADIANS(grids.grid_latitude - anchors.anchor_latitude) / 2.0), 2)
                    + COS(RADIANS(anchors.anchor_latitude))
                    * COS(RADIANS(grids.grid_latitude))
                    * POWER(SIN(RADIANS(grids.grid_longitude - anchors.anchor_longitude) / 2.0), 2)
                ))) AS distance_km
            FROM grid_centroids AS grids
            JOIN anchors USING (device_key)
        ),
        anchor_profile AS (
            SELECT
                device_key,
                COUNT(*)::BIGINT AS number_of_grids_visited,
                MIN(distance_km) AS anchor_grid_distance_min_km,
                QUANTILE_CONT(distance_km, 0.5)
                    AS anchor_grid_distance_p50_km,
                QUANTILE_CONT(distance_km, 0.75)
                    AS anchor_grid_distance_p75_km,
                QUANTILE_CONT(distance_km, 0.9)
                    AS anchor_grid_distance_p90_km,
                MAX(distance_km) AS anchor_grid_distance_max_km
            FROM anchor_distances
            GROUP BY device_key
        ),
        devices AS (
            SELECT DISTINCT device_key FROM base
        )
        SELECT
            devices.device_key AS device_id,
            daily_profile.n_active_days,
            daily_profile.median_daily_distance_km,
            daily_profile.mean_daily_distance_km,
            daily_profile.p90_daily_distance_km,
            daily_profile.sd_daily_distance_km,
            trajectory_profile.mean_stay_points_per_trajectory,
            trajectory_profile.min_stay_points_per_trajectory,
            trajectory_profile.max_stay_points_per_trajectory,
            trajectory_profile.p90_stay_points_per_trajectory,
            trip_shares.share_trip_100m_2km,
            trip_shares.share_trip_2_5km,
            trip_shares.share_trip_5_10km,
            trip_shares.share_trip_10_30km,
            trip_shares.share_trip_30km_plus,
            COALESCE(anchor_profile.number_of_grids_visited, 0)::BIGINT
                AS number_of_grids_visited,
            anchor_profile.anchor_grid_distance_min_km,
            anchor_profile.anchor_grid_distance_p50_km,
            anchor_profile.anchor_grid_distance_p75_km,
            anchor_profile.anchor_grid_distance_p90_km,
            anchor_profile.anchor_grid_distance_max_km
        FROM devices
        LEFT JOIN daily_profile USING (device_key)
        LEFT JOIN trajectory_profile USING (device_key)
        LEFT JOIN trip_shares USING (device_key)
        LEFT JOIN anchor_profile USING (device_key)
        ORDER BY device_id
    """
    return dataset._relation.query(source_alias, query)


def build_device_features(
    dataset: Dataset,
    *,
    device_id_col: str = "device_id",
    trajectory_id_col: str = "trajectory_id",
    trajectory_date_col: str = "trajectory_date",
    trajectory_distance_col: str = "trajectory_distance_m",
    visit_order_col: str = "visit_order",
    stay_point_col: str = "stay_point",
    grid_col: str = "spatial_cluster_label",
    lat_col: str = "lat",
    lon_col: str = "lon",
    trajectory_start_col: str = "trajectory_start_time",
) -> Dataset:
    """Build the twenty device features used by the Gurugram clustering workflow.

    The input is one row per visit in a trajectory-visit table. A trajectory's
    distance and stay-point count are calculated once per ``trajectory_id``;
    this avoids weighting trajectories by their number of visits. Daily
    distance statistics are then calculated from those deduplicated
    trajectories. The home-start anchor is the earliest valid visit for each
    device, and anchor distances are measured to the centroid of each visited
    grid using great-circle kilometres.

    Args:
        dataset: Lazy DuckDB-backed trajectory-visit Dataset.
        device_id_col: Device identifier column.
        trajectory_id_col: Unique trajectory identifier within each device.
        trajectory_date_col: Calendar date for each trajectory.
        trajectory_distance_col: Total trajectory distance in metres. Values
            repeated on every visit are intentionally used once per trajectory.
        visit_order_col: Visit order within a trajectory.
        stay_point_col: Binary stay-point indicator on visit rows.
        grid_col: Visited spatial-grid or spatial-cluster label.
        lat_col: WGS84 latitude of the visited grid/visit.
        lon_col: WGS84 longitude of the visited grid/visit.
        trajectory_start_col: Trajectory start timestamp used to choose the
            home-start anchor.

    Returns:
        A lazy Dataset with one row per device and ``device_id`` plus all
        :data:`DEVICE_CLUSTER_FEATURES` in the declared order. Fetching or
        clustering the result executes the DuckDB aggregation; source files are
        not modified.

    Raises:
        TypeError: If the input is not a Dataset or a column name is invalid.
        ValueError: If a required trajectory-visit column is missing.
        duckdb.Error: If DuckDB cannot evaluate the aggregation.
    """
    _validate_dataset(dataset)
    columns = (
        device_id_col,
        trajectory_id_col,
        trajectory_date_col,
        trajectory_distance_col,
        visit_order_col,
        stay_point_col,
        grid_col,
        lat_col,
        lon_col,
        trajectory_start_col,
    )
    for column in columns:
        _quote_identifier(column)
    _require_columns(dataset, columns)
    relation = _trajectory_feature_relation(
        dataset,
        device_id_col=device_id_col,
        trajectory_id_col=trajectory_id_col,
        trajectory_date_col=trajectory_date_col,
        trajectory_distance_col=trajectory_distance_col,
        visit_order_col=visit_order_col,
        stay_point_col=stay_point_col,
        grid_col=grid_col,
        lat_col=lat_col,
        lon_col=lon_col,
        trajectory_start_col=trajectory_start_col,
    )
    result = dataset._clone_with_relation(relation)
    result._device_feature_columns = DEVICE_CLUSTER_FEATURES
    return result


def _result_dataset(
    source: Dataset,
    writer: duckdb.DuckDBPyConnection,
    table_name: str,
    frame: pd.DataFrame,
) -> Dataset:
    """Store a small result frame in DuckDB and wrap it as a Dataset."""
    from opengeomove.data_loading.dataset import Dataset

    view_name = f"_ogm_device_result_view_{next(_OUTPUT_IDS)}"
    writer.register(view_name, frame)
    try:
        writer.execute(
            f"CREATE TEMP TABLE {_quote_identifier(table_name)} AS "
            f"SELECT * FROM {_quote_identifier(view_name)}"
        )
    finally:
        writer.unregister(view_name)
    result = Dataset(
        relation=writer.table(table_name),
        connection=writer,
        source_path=source.source,
        data_format=source.format,
    )
    result._filters = source.filters
    return result


@dataclass(frozen=True)
class DeviceClusterResult:
    """KMeans assignments and compact cluster-profile outputs.

    Attributes:
        assignments: One row per clustered device with ``cluster_id``,
            ``cluster_name``, and distance to the assigned centroid.
        profiles: Raw mean feature values per cluster plus ``device_count``.
        normalized_profiles: Per-feature min–max normalized cluster means,
            matching the heatmap convention in the project-update slide.
        feature_columns: Ordered features used by KMeans.
        model: Fitted scikit-learn KMeans model.
        preprocessor: Fitted imputation and scaling pipeline.
        metadata: Reproducibility settings and cluster counts.

    All three tabular outputs remain DuckDB-backed Datasets. The model and
    preprocessor are retained in memory so callers can inspect centers or
    reproduce assignments; source files remain read-only.
    """

    assignments: Dataset
    profiles: Dataset
    normalized_profiles: Dataset
    feature_columns: tuple[str, ...]
    model: KMeans
    preprocessor: Pipeline
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        """Return the number of clusters fitted by KMeans."""
        return int(self.model.n_clusters)

    def plot_profiles(
        self,
        *,
        figsize: tuple[float, float] = (14.0, 8.0),
        cmap: str = "viridis",
        show: bool = False,
    ) -> Any:
        """Plot cluster sizes beside the normalized feature-profile heatmap.

        The heatmap uses :attr:`normalized_profiles`, whose feature values are
        min–max normalized across clusters. The bar chart uses the recorded
        ``device_count`` in :attr:`profiles`; neither panel recomputes or
        re-reads the original trajectory source.

        Args:
            figsize: Matplotlib figure size in inches as ``(width, height)``.
            cmap: Matplotlib colormap used for the heatmap.
            show: If true, call ``matplotlib.pyplot.show()`` before returning.

        Returns:
            The Matplotlib Figure containing the bar chart, heatmap, and
            heatmap colorbar. Call ``savefig`` on the returned figure to write
            an image without displaying it.

        Raises:
            TypeError: If ``figsize``, ``cmap``, or ``show`` has an invalid
                type.
            ValueError: If ``figsize`` is not positive in both dimensions.
            ImportError: If Matplotlib is unavailable.
        """
        if (
            isinstance(figsize, (str, bytes))
            or not isinstance(figsize, tuple)
            or len(figsize) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in figsize)
        ):
            raise TypeError("figsize must be a (width, height) tuple of numbers")
        if figsize[0] <= 0 or figsize[1] <= 0:
            raise ValueError("figsize values must be positive")
        if not isinstance(cmap, str) or not cmap:
            raise TypeError("cmap must be a non-empty string")
        if not isinstance(show, bool):
            raise TypeError("show must be a bool")

        import matplotlib.pyplot as plt

        columns = tuple(self.normalized_profiles.columns)
        profile_rows = list(self.normalized_profiles.rows)
        profile_frame = pd.DataFrame(profile_rows, columns=columns)
        if profile_frame.empty:
            raise ValueError("No cluster profiles are available to plot")
        missing = [column for column in ("cluster_id", "device_count", *self.feature_columns) if column not in profile_frame]
        if missing:
            raise ValueError(f"Cluster profiles are missing plotting columns: {missing!r}")

        cluster_ids = profile_frame["cluster_id"].astype(int).to_numpy()
        feature_matrix = profile_frame.loc[:, self.feature_columns].to_numpy(dtype=float)
        counts = profile_frame["device_count"].astype(int).to_numpy()
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(float(figsize[0]), float(figsize[1])),
            gridspec_kw={"width_ratios": (1.0, 1.8)},
        )
        size_axes, heatmap_axes = axes
        size_axes.bar(cluster_ids, counts, color="#3f7896")
        size_axes.set_title("Device count by cluster")
        size_axes.set_xlabel("Cluster")
        size_axes.set_ylabel("Number of devices")
        size_axes.set_xticks(cluster_ids)

        image = heatmap_axes.imshow(
            feature_matrix,
            aspect="auto",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )
        heatmap_axes.set_title(
            "Home-start anchor-spread clustering: normalized cluster profile"
        )
        heatmap_axes.set_xlabel("Cluster")
        heatmap_axes.set_ylabel("Feature")
        heatmap_axes.set_xticks(np.arange(len(cluster_ids)))
        heatmap_axes.set_xticklabels(cluster_ids)
        heatmap_axes.set_yticks(np.arange(len(self.feature_columns)))
        heatmap_axes.set_yticklabels(self.feature_columns)
        heatmap_axes.tick_params(axis="y", labelsize=8)
        colorbar = figure.colorbar(image, ax=heatmap_axes)
        colorbar.set_label("Min-max normalized cluster mean")
        figure.suptitle("Device clustering")
        figure.tight_layout()
        if show:
            plt.show()
        return figure


def cluster_devices(
    dataset: Dataset,
    *,
    device_id_col: str = "device_id",
    feature_columns: Sequence[str] | None = None,
    n_clusters: int = 5,
    scale: str = "standard",
    missing: str = "median",
    random_state: int | None = 42,
    n_init: int = 10,
    labels: Mapping[int, str] | None = None,
) -> DeviceClusterResult:
    """Fit deterministic KMeans clusters to device mobility features.

    ``dataset`` should normally be the result of
    :func:`build_device_features`, although any one-row-per-device Dataset
    containing the same named columns is accepted. DuckDB projects and casts
    only the device identifier and selected features before the device-level
    matrix is materialized for scikit-learn.

    Args:
        dataset: Device-feature Dataset with one row per device.
        device_id_col: Device identifier column.
        feature_columns: Ordered numeric columns. Defaults to all twenty
            :data:`DEVICE_CLUSTER_FEATURES` from the slide.
        n_clusters: Number of KMeans clusters; defaults to five.
        scale: Feature scaling, one of ``"standard"``, ``"robust"``, or
            ``"none"``. Standard scaling is applied before KMeans by default.
        missing: Missing-feature policy, ``"median"`` or ``"drop"``.
        random_state: KMeans seed. The default makes the result reproducible.
        n_init: Number of KMeans initializations.
        labels: Optional mapping from numeric cluster IDs to names. When
            omitted, names from the project-update slide are attached. Cluster
            IDs remain model labels and should not be interpreted as ordered.

    Returns:
        A :class:`DeviceClusterResult` containing lazy assignment/profile
        Datasets and the fitted KMeans objects. KMeans necessarily materializes
        the projected device-by-feature matrix, but never materializes the
        complete trajectory-visit source in Python.

    Raises:
        TypeError: If the Dataset, column names, or integer parameters are
            invalid.
        ValueError: If columns/settings are invalid, device IDs are duplicated,
            too few usable devices are available, or a feature is non-numeric.
    """
    _validate_dataset(dataset)
    _quote_identifier(device_id_col)
    _require_columns(dataset, (device_id_col,))

    if feature_columns is None:
        selected = DEVICE_CLUSTER_FEATURES
    else:
        if isinstance(feature_columns, str):
            raise TypeError("feature_columns must be a sequence of column names")
        selected = tuple(feature_columns)
    if not selected:
        raise ValueError("feature_columns must contain at least one column")
    if len(set(selected)) != len(selected):
        raise ValueError("feature_columns must not contain duplicates")
    for column in selected:
        _quote_identifier(column)
    _require_columns(dataset, selected)

    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int):
        raise TypeError("n_clusters must be an integer")
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")
    if isinstance(n_init, bool) or not isinstance(n_init, int):
        raise TypeError("n_init must be an integer")
    if n_init < 1:
        raise ValueError("n_init must be at least 1")
    if scale not in {"standard", "robust", "none"}:
        raise ValueError("scale must be 'standard', 'robust', or 'none'")
    if missing not in {"median", "drop"}:
        raise ValueError("missing must be 'median' or 'drop'")
    if labels is not None and not isinstance(labels, Mapping):
        raise TypeError("labels must be a mapping from cluster IDs to names")

    source_alias = f"_ogm_device_cluster_source_{next(_OUTPUT_IDS)}"
    projection = ", ".join(
        [_quote_identifier(device_id_col) + " AS device_id"]
        + [
            f"TRY_CAST({_quote_identifier(column)} AS DOUBLE) AS {_quote_identifier(column)}"
            for column in selected
        ]
    )
    relation = dataset._relation.query(
        source_alias,
        f"SELECT {projection} FROM {source_alias} "
        "WHERE " + _quote_identifier(device_id_col) + " IS NOT NULL",
    )
    data = relation.df()
    if data.empty:
        raise ValueError("No non-null device rows are available for clustering")
    if data["device_id"].duplicated().any():
        raise ValueError("The device-feature Dataset must contain one row per device")

    numeric = data.loc[:, selected].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.mask(~np.isfinite(numeric))
    if missing == "drop":
        keep = numeric.notna().all(axis=1)
        data = data.loc[keep].reset_index(drop=True)
        numeric = numeric.loc[keep].reset_index(drop=True)
    if len(data) < n_clusters:
        raise ValueError(
            f"Need at least {n_clusters} usable devices to create "
            f"{n_clusters} clusters; got {len(data)}"
        )

    imputer: Any
    if missing == "median":
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    else:
        imputer = "passthrough"
    scaler: Any
    if scale == "standard":
        scaler = StandardScaler()
    elif scale == "robust":
        scaler = RobustScaler()
    else:
        scaler = "passthrough"
    preprocessor = Pipeline([("imputer", imputer), ("scaler", scaler)])
    matrix = preprocessor.fit_transform(numeric)
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels_array = model.fit_predict(matrix).astype(int)
    distances = model.transform(matrix)[np.arange(len(labels_array)), labels_array]

    label_map = DEVICE_CLUSTER_LABELS if labels is None else labels
    cluster_names = [str(label_map.get(int(label), label)) for label in labels_array]
    assignments = pd.DataFrame(
        {
            "device_id": data["device_id"].astype(str),
            "cluster_id": labels_array,
            "cluster_name": cluster_names,
            "distance": distances,
        }
    )

    imputer_step = preprocessor.named_steps["imputer"]
    profile_values = (
        imputer_step.transform(numeric)
        if imputer_step != "passthrough"
        else numeric.to_numpy(dtype=float)
    )
    profile_data = pd.DataFrame(profile_values, columns=selected)
    profile_data["cluster_id"] = labels_array
    profile_data = profile_data.groupby("cluster_id", sort=True)[list(selected)].mean()
    profile_data = profile_data.reindex(range(n_clusters))
    profile_output = profile_data.reset_index()
    profile_output.insert(
        1,
        "cluster_name",
        [str(label_map.get(index, index)) for index in range(n_clusters)],
    )
    profile_output.insert(
        2,
        "device_count",
        [int(np.sum(labels_array == index)) for index in range(n_clusters)],
    )

    normalized = profile_output.copy()
    for column in selected:
        values = normalized[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            normalized[column] = 0.0
            continue
        minimum = float(np.nanmin(values[finite]))
        maximum = float(np.nanmax(values[finite]))
        if math.isclose(minimum, maximum):
            normalized[column] = 0.0
        else:
            normalized[column] = (values - minimum) / (maximum - minimum)
    normalized = normalized.fillna(0.0)

    writer = dataset._connection.cursor()
    suffix = next(_OUTPUT_IDS)
    assignments_dataset = _result_dataset(
        dataset, writer, f"_ogm_device_assignments_{suffix}", assignments
    )
    profiles_dataset = _result_dataset(
        dataset, writer, f"_ogm_device_profiles_{suffix}", profile_output
    )
    normalized_dataset = _result_dataset(
        dataset, writer, f"_ogm_device_profiles_normalized_{suffix}", normalized
    )
    return DeviceClusterResult(
        assignments=assignments_dataset,
        profiles=profiles_dataset,
        normalized_profiles=normalized_dataset,
        feature_columns=tuple(selected),
        model=model,
        preprocessor=preprocessor,
        metadata={
            "n_clusters": n_clusters,
            "scale": scale,
            "missing": missing,
            "random_state": random_state,
            "n_init": n_init,
            "feature_columns": list(selected),
            "device_count": len(assignments),
        },
    )


__all__ = [
    "DEVICE_CLUSTER_FEATURES",
    "DEVICE_CLUSTER_LABELS",
    "DeviceClusterResult",
    "build_device_features",
    "cluster_devices",
]
