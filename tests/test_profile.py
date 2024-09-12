import polars as pl
from polars_profile import profile


# Create a test function for the profile function using a dataframe containing each valid datatype
def test_profile():
    # Create a dataframe
    df = pl.DataFrame(
        {
            "workout_id": [i for i in range(500)],
            "slug": [f"slug_{i}" for i in range(500)],
            "activity_weight": [i % 10 for i in range(500)],
            "google_activity": [f"activity_{i}" for i in range(500)],
            "activity_mfp": [1.0e14 + i for i in range(500)],
            "classification": [f"class_{i}" for i in range(500)],
            "timing_style": [f"style_{i}" for i in range(500)],
            "recommended_age": [i % 6 for i in range(500)],
            "recommended_workout_lengths": [f"length_{i}" for i in range(500)],
            "recommended_circuits": [i % 2 for i in range(500)],
        }
    )
    # Profile the dataframe
    pdf = df.profile()
    # Assert the profiled dataframe has the correct shape
    assert pdf.shape == (10, 11)
    # Assert the profiled dataframe has the correct columns
    assert pdf.columns == [
        "column",
        "mean",
        "min",
        "max",
        "median",
        "std",
        "mean_length",
        "min_length",
        "max_length",
        "count_null",
        "count",
    ]
    # Assert the profiled dataframe has the correct datatypes
    assert pdf.dtypes == [
        pl.String,
        pl.Float64,
        pl.Float64,
        pl.Float64,
        pl.Float64,
        pl.Float64,
        pl.UInt32,
        pl.UInt32,
        pl.UInt32,
        pl.UInt32,
        pl.UInt32,
    ]
    # Assert the profiled dataframe has the correct values
    df_dict = pdf.to_dict()
    assert list(df_dict.keys()) == ['column', 'mean', 'min', 'max', 'median', 'std', 'mean_length', 'min_length', 'max_length', 'count_null', 'count']
    assert list(df_dict['column']) == [
    "workout_id",
    "slug",
    "activity_weight",
    "google_activity",
    "activity_mfp",
    "classification",
    "timing_style",
    "recommended_age",
    "recommended_workout_lengths",
    "recommended_circuits",
    ]
    assert list(df_dict["mean"]) == [
        249.5,
        None,
        4.5,
        None,
        100000000000249.47,
        None,
        None,
        2.492,
        None,
        0.5
    ]

    assert list(df_dict["min"]) == [
        0.0,
        None,
        0.0,
        None,
        1.0e14,
        None,
        None,
        0.0,
        None,
        0.0
    ]

    assert list(df_dict["max"]) == [
        499.0,
        None,
        9.0,
        None,
        1.0e14 + 499,
        None,
        None,
        5.0,
        None,
        1.0
    ]

    assert list(df_dict["median"]) == [
        249.5,
        None,
        4.5,
        None,
        100000000000249.5,
        None,
        None,
        2.0,
        None,
        0.5
    ]

    assert list(df_dict["std"]) == [
        144.33727862198316,
        None,
        2.8722813232690143,
        None,
        144.33728200490162,
        None,
        None,
        1.70936713435119,
        None,
        0.5
    ]

    assert list(df_dict["mean_length"]) == [
        None,
        7,
        None,
        11,
        None,
        8,
        8,
        None,
        9,
        None
    ]

    assert list(df_dict["min_length"]) == [
        None,
        6,
        None,
        10,
        None,
        7,
        7,
        None,
        8,
        None
    ]

    assert list(df_dict["max_length"]) == [
        None,
        8,
        None,
        12,
        None,
        9,
        9,
        None,
        10,
        None
    ]

    assert list(df_dict["count_null"]) == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    assert list(df_dict["count"]) == [
        500,
        500,
        500,
        500,
        500,
        500,
        500,
        500,
        500,
        500,
    ]
