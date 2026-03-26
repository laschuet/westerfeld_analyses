import numpy as np
import pandas as pd

from collections.abc import Iterable

EXPERIMENT_COLUMNS = [
    "Experimental_Year",
    "Date",
    "Crop",
    "Replicate",
    "Tillage",
    "Fertilization",
    "Treatment",
    "Habitat",
    "Beneficial",
    "Precrop",
]


def resolve_filter(values, df, column):
    if values is None:
        return list(np.unique(df[column]))
    if isinstance(values, (int, np.integer, str)):
        return [values]
    return values


def postprocess_label(label):
    return label.replace(" ", "").lower()


def create_label(values):
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        values = [values]

    if len(values) == 1:
        label = str(values[0])
    else:
        label = "_".join(str(value) for value in values)

    return postprocess_label(label)


def debug(df, n=50):
    print(df.head(n))
    print(df.shape)
    print(df.dtypes)


def pivot(
    df,
    values,
    index,
    columns=None,
    aggfunc="sum",
    fill_value=0,
    index_axis_name="Sample",
    column_axis_name="Feature",
    drop=True,
):
    df = df.pivot_table(
        values=values,
        index=index,
        columns=columns,
        aggfunc=aggfunc,
        fill_value=fill_value,
    )
    df = df.reset_index(drop=drop)
    df = df.rename_axis(index_axis_name)
    df = df.rename_axis(column_axis_name, axis="columns")
    return df


def split(df, columns):
    return df[columns], df.drop(columns=columns)


def merge_and_unpivot(df_split_1, df_split_2, index, column, value_name, fill_value=0):
    df_piv = pd.merge(
        df_split_1, df_split_2, left_index=True, right_index=True, how="inner"
    )

    df_unpiv = df_piv.melt(id_vars=index, var_name=column, value_name=value_name)
    df_unpiv = df_unpiv[df_unpiv[value_name] != fill_value]

    return df_unpiv


def taxonomy_level(type_label):
    if type_label == "Fungi":
        return "Species"
    else:
        return "Genus"
