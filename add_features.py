import pandas as pd
import numpy as np

import keras

def from_date_get_dow_dom_doy(ser_date, init_col_name=""):
    """

    :param ser_date: pd.Series of type: date
    :return: pd.Dataframe with cols:
            "dow": day of week
            "dom": day of month
            "doy": day of year
    """
    df = pd.DataFrame()
    df[init_col_name+"dow"] = ser_date.dt.dayofweek
    df[init_col_name+"dom"] = ser_date.dt.day
    df[init_col_name+"doy"] = ser_date.dt.dayofyear

    return df

def from_values_to_values_diff(df_values):
    """

    :param df_values: df with numeric columns
    :return: diff columns corresponding to the df colomns
    """
    df = pd.DataFrame()
    diff_cols = ["diff_"+i for i in df_values.columns]
    df[diff_cols] = df_values.iloc.diff()

    return df

def from_shlomos_df_to_df_with_features(df, val_columns=[""]):
    """

    :param df: df from shlomo's csv files. including the columns:
                "Date": the datetime of the measure
    :return:
    """
    df.Date = pd.to_datetime(df.Date, dayfirst=True)

    vals_cols = ["UX1", "UX2", "UX3", "UX4", "UX5"]
    more_cols_to_diff = ["SP500"]
    cols_to_diff = val_columns + more_cols_to_diff
    df = pd.concat([df, from_values_to_values_diff(df[cols_to_diff])], axis=1)

def drop_rows_with_null_dates(df):
    print("drop nulls:")
    print("before:", df.shape, df.Date.isnull().sum())

    nulls_idx = df[df.Date.isnull()].index
    df.drop(index=nulls_idx, inplace=True)

    print("after:", df.shape, df.Date.isnull().sum())

    return df

