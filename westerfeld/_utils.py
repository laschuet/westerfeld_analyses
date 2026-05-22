import numpy as np

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


def taxonomy_level(type_label):
    if type_label == "Fungi":
        return "Species"
    else:
        return "Genus"


def calc_iou(d1, d2):
    union = set(d1 + d2)
    intersect = np.intersect1d(d1, d2)
    return len(intersect) / len(union)
