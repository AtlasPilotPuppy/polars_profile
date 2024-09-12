#![allow(clippy::unused_unit)]

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

struct ProfileResult {
    column: String,
    mean: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
    median: Option<f64>,
    std: Option<f64>,
    mean_length: Option<u32>,
    min_length: Option<u32>,
    max_length: Option<u32>,
    count_null: u32,
    count: u32,
}

fn min_series(series: &Series) -> Option<f64> {
    match series.dtype() {
        DataType::UInt8 => series.u8().unwrap().min().map(|v| v as f64),
        DataType::UInt16 => series.u16().unwrap().min().map(|v| v as f64),
        DataType::UInt32 => series.u32().unwrap().min().map(|v| v as f64),
        DataType::UInt64 => series.u64().unwrap().min().map(|v| v as f64),
        DataType::Int8 => series.i8().unwrap().min().map(|v| v as f64),
        DataType::Int16 => series.i16().unwrap().min().map(|v| v as f64),
        DataType::Int32 => series.i32().unwrap().min().map(|v| v as f64),
        DataType::Int64 => series.i64().unwrap().min().map(|v| v as f64),
        DataType::Float32 => series.f32().unwrap().min().map(|v| v as f64),
        DataType::Float64 => series.f64().unwrap().min(),
        _ => None,
    }
}

fn max_series(series: &Series) -> Option<f64> {
    match series.dtype() {
        DataType::UInt8 => series.u8().unwrap().max().map(|v| v as f64),
        DataType::UInt16 => series.u16().unwrap().max().map(|v| v as f64),
        DataType::UInt32 => series.u32().unwrap().max().map(|v| v as f64),
        DataType::UInt64 => series.u64().unwrap().max().map(|v| v as f64),
        DataType::Int8 => series.i8().unwrap().max().map(|v| v as f64),
        DataType::Int16 => series.i16().unwrap().max().map(|v| v as f64),
        DataType::Int32 => series.i32().unwrap().max().map(|v| v as f64),
        DataType::Int64 => series.i64().unwrap().max().map(|v| v as f64),
        DataType::Float32 => series.f32().unwrap().max().map(|v| v as f64),
        DataType::Float64 => series.f64().unwrap().max(),
        _ => None,
    }
}

fn mean_series(series: &Series) -> Option<f64> {
    match series.dtype() {
        DataType::UInt8 => series.u8().unwrap().mean().map(|v| v as f64),
        DataType::UInt16 => series.u16().unwrap().mean().map(|v| v as f64),
        DataType::UInt32 => series.u32().unwrap().mean().map(|v| v as f64),
        DataType::UInt64 => series.u64().unwrap().mean().map(|v| v as f64),
        DataType::Int8 => series.i8().unwrap().mean().map(|v| v as f64),
        DataType::Int16 => series.i16().unwrap().mean().map(|v| v as f64),
        DataType::Int32 => series.i32().unwrap().mean().map(|v| v as f64),
        DataType::Int64 => series.i64().unwrap().mean().map(|v| v as f64),
        DataType::Float32 => series.f32().unwrap().mean().map(|v| v as f64),
        DataType::Float64 => series.f64().unwrap().mean(),
        _ => None,
    }
}

fn median_series(series: &Series) -> Option<f64> {
    match series.dtype() {
        DataType::UInt8 => series.u8().unwrap().median().map(|v| v as f64),
        DataType::UInt16 => series.u16().unwrap().median().map(|v| v as f64),
        DataType::UInt32 => series.u32().unwrap().median().map(|v| v as f64),
        DataType::UInt64 => series.u64().unwrap().median().map(|v| v as f64),
        DataType::Int8 => series.i8().unwrap().median().map(|v| v as f64),
        DataType::Int16 => series.i16().unwrap().median().map(|v| v as f64),
        DataType::Int32 => series.i32().unwrap().median().map(|v| v as f64),
        DataType::Int64 => series.i64().unwrap().median().map(|v| v as f64),
        DataType::Float32 => series.f32().unwrap().median().map(|v| v as f64),
        DataType::Float64 => series.f64().unwrap().median(),
        _ => None,
    }
}

fn std_series(series: &Series) -> Option<f64> {
    match series.dtype() {
        // Assume whole population using degrees of freedom 0
        DataType::UInt8 => series.u8().unwrap().std(0).map(|v| v as f64),
        DataType::UInt16 => series.u16().unwrap().std(0).map(|v| v as f64),
        DataType::UInt32 => series.u32().unwrap().std(0).map(|v| v as f64),
        DataType::UInt64 => series.u64().unwrap().std(0).map(|v| v as f64),
        DataType::Int8 => series.i8().unwrap().std(0).map(|v| v as f64),
        DataType::Int16 => series.i16().unwrap().std(0).map(|v| v as f64),
        DataType::Int32 => series.i32().unwrap().std(0).map(|v| v as f64),
        DataType::Int64 => series.i64().unwrap().std(0).map(|v| v as f64),
        DataType::Float32 => series.f32().unwrap().std(0).map(|v| v as f64),
        DataType::Float64 => series.f64().unwrap().std(0),
        _ => None,
    }
}

fn str_lengths(series: &Series) -> Option<(u32, u32, u32)> {
    match series.dtype() {
        DataType::String => {
            let ca = series.str().ok()?;
            let lengths: Vec<Option<u32>> = ca
                .into_iter()
                .map(|opt_v| opt_v.map(|v| v.len() as u32))
                .collect();

            let max_len = lengths.iter().max().copied().unwrap_or_default();
            let min_len = lengths.iter().min().copied().unwrap_or_default();
            let mean_len =
                lengths.into_iter().map(|v| v.unwrap_or(0)).sum::<u32>() / ca.len() as u32;

            Some((mean_len, min_len.unwrap_or(0), max_len.unwrap_or(0)))
        },
        _ => None,
    }
}

fn series_stats(series: &Series) -> Option<ProfileResult> {
    match series.dtype() {
        // match numeric types
        DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Float32
        | DataType::Float64 => Some(ProfileResult {
            column: series.name().to_string(),
            mean: mean_series(series),
            min: min_series(series),
            max: max_series(series),
            median: median_series(series),
            std: std_series(series),
            mean_length: None,
            min_length: None,
            max_length: None,
            count_null: series.null_count() as u32,
            count: series.len() as u32,
        }),
        // match string types
        DataType::String => {
            // get the results from the str_lengths function
            let (mean_len, min_len, max_len) = str_lengths(series)?;
            Some(ProfileResult {
                column: series.name().to_string(),
                mean: None,
                min: None,
                max: None,
                median: None,
                std: None,
                mean_length: Some(mean_len),
                min_length: Some(min_len),
                max_length: Some(max_len),
                count_null: series.null_count() as u32,
                count: series.len() as u32,
            })
        },
        DataType::Struct(_) => None,
        _ => None,
    }
}

fn output_fields(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "profile_result".into(),
        DataType::Struct(vec![
            Field::new("column".into(), DataType::String),
            Field::new("mean".into(), DataType::Float64),
            Field::new("min".into(), DataType::Float64),
            Field::new("max".into(), DataType::Float64),
            Field::new("median".into(), DataType::Float64),
            Field::new("std".into(), DataType::Float64),
            Field::new("mean_length".into(), DataType::UInt32),
            Field::new("min_length".into(), DataType::UInt32),
            Field::new("max_length".into(), DataType::UInt32),
            Field::new("count_null".into(), DataType::UInt32),
            Field::new("count".into(), DataType::UInt32),
        ]),
    ))
}

fn profile_internal(inputs: &[Series]) -> PolarsResult<Series> {
    let mut columns = vec![];
    let mut means = vec![];
    let mut mins = vec![];
    let mut maxs = vec![];
    let mut medians = vec![];
    let mut stds = vec![];
    let mut mean_lengths = vec![];
    let mut min_lengths = vec![];
    let mut max_lengths = vec![];
    let mut count_nulls = vec![];
    let mut counts = vec![];

    for series in inputs {
        if let Some(result) = series_stats(series) {
            columns.push(result.column);
            means.push(result.mean);
            mins.push(result.min);
            maxs.push(result.max);
            medians.push(result.median);
            stds.push(result.std);
            mean_lengths.push(result.mean_length);
            min_lengths.push(result.min_length);
            max_lengths.push(result.max_length);
            count_nulls.push(result.count_null);
            counts.push(result.count);
        }
    }

    let df = DataFrame::new(vec![
        Series::new("column".into(), columns),
        Series::new("mean".into(), means),
        Series::new("min".into(), mins),
        Series::new("max".into(), maxs),
        Series::new("median".into(), medians),
        Series::new("std".into(), stds),
        Series::new("mean_length".into(), mean_lengths),
        Series::new("min_length".into(), min_lengths),
        Series::new("max_length".into(), max_lengths),
        Series::new("count_null".into(), count_nulls),
        Series::new("count".into(), counts),
    ])?;

    Ok(df.into_struct("profile_result".into()).into_series())
}

#[polars_expr(output_type_func=output_fields)]
fn profile(inputs: &[Series]) -> PolarsResult<Series> {
    profile_internal(inputs)
}

mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_profile_numeric_columns() {
        let series1 = Series::new("int32_col".into(), vec![1, 2, 3, 4, 5]);
        let series2 = Series::new("float64_col".into(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let result = profile_internal(&[series1, series2]).unwrap();

        // Check the structure and values of the result
        assert_eq!(result.name(), "profile_result");
        assert_eq!(
            result.dtype(),
            &DataType::Struct(vec![
                Field::new("column".into(), DataType::String),
                Field::new("mean".into(), DataType::Float64),
                Field::new("min".into(), DataType::Float64),
                Field::new("max".into(), DataType::Float64),
                Field::new("median".into(), DataType::Float64),
                Field::new("std".into(), DataType::Float64),
                Field::new("mean_length".into(), DataType::UInt32),
                Field::new("min_length".into(), DataType::UInt32),
                Field::new("max_length".into(), DataType::UInt32),
                Field::new("count_null".into(), DataType::UInt32),
                Field::new("count".into(), DataType::UInt32),
            ])
        );

        // Further checks can be added to verify the values within the struct
    }

    #[test]
    fn test_profile_string_columns() {
        let series1 = Series::new("string_col".into(), vec!["a", "bb", "ccc", "dddd", "eeeee"]);

        let result = profile_internal(&[series1]).unwrap();

        // Check the structure and values of the result
        assert_eq!(result.name(), "profile_result");
        assert_eq!(
            result.dtype(),
            &DataType::Struct(vec![
                Field::new("column".into(), DataType::String),
                Field::new("mean".into(), DataType::Float64),
                Field::new("min".into(), DataType::Float64),
                Field::new("max".into(), DataType::Float64),
                Field::new("median".into(), DataType::Float64),
                Field::new("std".into(), DataType::Float64),
                Field::new("mean_length".into(), DataType::UInt32),
                Field::new("min_length".into(), DataType::UInt32),
                Field::new("max_length".into(), DataType::UInt32),
                Field::new("count_null".into(), DataType::UInt32),
                Field::new("count".into(), DataType::UInt32),
            ])
        );

        // Further checks can be added to verify the values within the struct
    }
}
