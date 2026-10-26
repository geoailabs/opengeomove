"""Traffic analysis zone creation, population attachment, and comparison.

TAZ operations use GeoDataFrames for polygon input/output because OpenGeoMove's
shared :class:`Dataset` currently represents DuckDB-backed tabular files.
"""

from opengeomove.taz.automatic import AutomaticTAZConfig, create_automatic_taz
from opengeomove.taz.comparison import TAZComparison, compare_taz
from opengeomove.taz.config import TAZConfig
from opengeomove.taz.create import TAZBuilder, create, create_taz
from opengeomove.taz.exceptions import TAZDependencyError, TAZError, TAZGeometryError
from opengeomove.taz.models import StudyArea, TAZFrame, TAZReport
from opengeomove.taz.plotting import plot_taz
from opengeomove.taz.population import PopulationSource, attach_population, population_cells
from opengeomove.taz.socioeconomic import (
    SOCIOECONOMIC_CLASSES,
    SOCIOECONOMIC_FEATURE_COLUMNS,
    SocioeconomicClassifier,
    extract_socioeconomic_features,
    fit_socioeconomic_classifier,
)
from opengeomove.data_loading.google_buildings import download_open_buildings_25d

__all__ = [
    "AutomaticTAZConfig",
    "PopulationSource",
    "SOCIOECONOMIC_CLASSES",
    "SOCIOECONOMIC_FEATURE_COLUMNS",
    "SocioeconomicClassifier",
    "StudyArea",
    "TAZBuilder",
    "TAZComparison",
    "TAZConfig",
    "TAZDependencyError",
    "TAZError",
    "TAZFrame",
    "TAZGeometryError",
    "TAZReport",
    "attach_population",
    "compare_taz",
    "create",
    "create_automatic_taz",
    "create_taz",
    "download_open_buildings_25d",
    "extract_socioeconomic_features",
    "fit_socioeconomic_classifier",
    "population_cells",
    "plot_taz",
]
