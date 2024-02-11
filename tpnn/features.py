from functools import partial
from typing import Callable
import numpy as np
import pandas as pd


def _inplace_modify(
    old_df: pd.DataFrame, aux_df: pd.DataFrame, inplace: bool
) -> pd.DataFrame | None:
    if inplace:
        old_df[aux_df.columns] = aux_df
        return

    return old_df.assign(**{column: aux_df[column] for column in aux_df.columns})


def centerize(features: pd.DataFrame, inplace: bool = False) -> pd.DataFrame | None:
    numerics = features.select_dtypes(include=np.number)
    minmax = numerics.agg(["min", "max"])
    centers = minmax.mean(axis=0)
    numerics -= centers

    return _inplace_modify(features, numerics, inplace)


def normalize(features: pd.DataFrame, inplace: bool = False) -> pd.DataFrame | None:
    numerics = features.select_dtypes(include=np.number)
    maxs = numerics.max(axis=0)
    maxs = maxs[maxs > 0]
    numerics /= maxs

    return _inplace_modify(features, numerics, inplace)


def replace_categorical_values(
    features: pd.DataFrame, inplace: bool = False
) -> pd.DataFrame | None:
    def convert2codes(column: "pd.Series[pd.Categorical]") -> int:
        return column.cat.codes

    categorical = features.select_dtypes(include=object)
    categorical = categorical.astype("category").aggregate(convert2codes)

    return _inplace_modify(features, categorical, inplace)


def remove_outliers_zscore(
    features: pd.DataFrame, zscore_thresh: int = 3, inplace: bool = False
) -> pd.DataFrame:
    numerics = features.select_dtypes(include=np.number)
    means = numerics.mean(axis=0)
    stds = numerics.std(axis=0)
    exclude = (((numerics - means) / stds).abs() >= zscore_thresh).any(axis=1)
    index = exclude[exclude].index

    return features.drop(index=index, inplace=inplace)


def remove_strong_relations(
    features: pd.DataFrame,
    corr2exclude_index: Callable[[pd.DataFrame], "pd.Series[bool]"],
    thresh: float = 0.5,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Убрать из датафрейма `features` сильно скоррелированные колонки.

    1) Построить матрицу корреляций для числовых фичей.
    2) Убрать диагональные единицы.
    3) Применить функцию выбора названий колонок `corr2exclude_series`.
    4) Вернуть часть `features` с нужными колонками.

    Args:
        - features (pandas.DataFrame): Фрейм с признаками.
            Могут быть и категориальными, и числовыми.
        - corr2exclude_series((pandas.DataFrame) -> pandas.Series[bool]): Инверсия управления
            выбора ненужных колонок из `features`.
        - thresh(float[0.0, 1.0]): Число от 0 до 1.
            Порог по которому определяется являются ли фичи скоррелированными.

    Returns:
        pandas.DataFrame: Фрейм только теми колонками, которые нескоррелированы.
    """
    correlation_matrix = features.corr(numeric_only=True)

    diagonal = np.eye(correlation_matrix.shape[0], dtype=bool)
    correlated_matrix = correlation_matrix.mask(diagonal).abs() >= thresh

    exclude = corr2exclude_index(correlated_matrix)
    exclude = exclude[exclude].index

    print(f"Excluding columns {', '.join(exclude.to_list())} due to strong correlation")
    # include = features.columns.map(lambda feature_name: feature_name not in exclude)

    return features.drop(
        columns=exclude,
        inplace=inplace,
    )


def remove_na(features: pd.DataFrame, inplace: bool = False) -> pd.DataFrame | None:
    print(features.isna().any(axis=1))


def _have_any_relation(corr: pd.DataFrame) -> "pd.Series[bool]":
    """Убирать признак, если он скоррелирован хотя бы с одним другим признаком."""
    exclude = corr.any(axis=1)
    return exclude


def _all_except_one(corr: pd.DataFrame) -> "pd.Series[bool]":
    """
    1) Выбрать строку с наибольшей суммой (если таких несколько, то выбираем любую).
    2) Удалить её и занулить соответсвующий столбец.
    3) Повторять предыдущие два шага, пока сумма по таблице не будет равна 0.
    """

    corr_copy = corr.copy()

    def column_key(row: "pd.Index[str]") -> int:
        return corr_copy[row].sum()

    while corr_copy.sum().sum() > 0:
        column: str = max(corr_copy.index, key=column_key)
        corr_copy.drop(columns=column, index=column, inplace=True)

    exclude = corr.index.to_series().map(
        lambda feature_name: feature_name not in corr_copy.index
    )
    return exclude


rsr_any = partial(
    remove_strong_relations,
    corr2exclude_index=_have_any_relation,
)

rsr_except_one = partial(
    remove_strong_relations,
    corr2exclude_index=_all_except_one,
)
