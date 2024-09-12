from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List

import polars as pl
from polars.plugins import register_plugin_function

from polars_profile._internal import __version__ as __version__

if TYPE_CHECKING:
    from polars_profile.typing import IntoExpr

LIB = Path(__file__).parent


def profile(df: pl.DataFrame) -> pl.DataFrame:
    # Convert dataframe to list of series
    series_list = [df.to_series(i) for i in range(len(df.columns))]
    return register_plugin_function(
        args=series_list,
        plugin_path=LIB,
        function_name="profile",
        is_elementwise=False,
    )

# Add this new function
def _df_profile(df: pl.DataFrame) -> pl.DataFrame:
    pdf = df.select( profile(df).alias("profile"))
    profile_series = pdf.to_series().struct.unnest()
    return pl.DataFrame(profile_series)

# Monkey patch the DataFrame class
pl.DataFrame.profile = _df_profile
