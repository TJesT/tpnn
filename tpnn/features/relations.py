import pandas as pd
import numpy as np

from typing import Literal, override
from ..core.pipable_base import Pipeable

type column_name = str


def _filter_if_any(corr: pd.DataFrame) -> "pd.Series[column_name]":
    """
    Получить некоррелируюшие признаки за счёт удаления всех, что
    скоррелированы хотя бы с одним другим признаком.
    """
    include = ~corr.all(axis=1)
    return include[include].index


def _filter_sequentially(corr: pd.DataFrame) -> "pd.Series[column_name]":
    """
    Получить некоррелирующие признаки за счёт последовательного удаления
    наиболее скоррелированных.

    1) Выбрать строку с наибольшей суммой (если таких несколько, то первую).
    2) Удалить строку и столбец с найденным индексом.
    3) Повторять предыдущие два шага, пока сумма по таблице не будет равна 0.
    """

    def column_sum(row: "pd.Index[column_name]") -> int:
        return corr[row].sum()

    while corr.values.sum() > 0:
        column: column_name = max(corr.index, key=column_sum)
        corr.drop(columns=column, index=column, inplace=True)

    return corr.index


class CorelationFilter(Pipeable):
    """
    Класс для удаления скоррелированных признаков по матрице ковариаций.
    Линейно скоррелированными будут считаться признаки, для которых выполняется
    `|cov(Xᵢ, Xⱼ)| >= thresh`

    Args:
        `thresh`: float[0.0, 1.0]
            Число от 0 до 1. Порог по которому определяется являются ли фичи\
            скоррелированными.
        `strategy`: Literal['all', 'seq']
            -`'all'`: Удалить все признаки, что скоррелированы хотя бы с одним другим.
            -`'seq'`: Последовательно удалять признак, больше всего влияющий на остальные.

    """

    __strategy_mapping = {
        "all": _filter_if_any,
        "seq": _filter_sequentially,
    }

    def __init__(
        self,
        thresh: float = 0.5,
        strategy: Literal["all", "seq"] = "all",
    ) -> None:
        self.filter = self.__strategy_mapping.get(strategy)
        self.thresh = thresh

    @override
    def __call__(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Убрать из датафрейма `features` сильно скоррелированные колонки.

        1) Построить матрицу корреляций для числовых фичей.
        2) Убрать диагональные единицы.
        3) Применить функцию выбора колонок `corr2exclude_series`.
        4) Вернуть часть `features` с нужными колонками.

        Args:
            `features`: pd.DataFrame
                Фрейм с признаками. Могут быть и категориальными, и числовыми.

        Returns:
            Фрейм только c теми колонками, которые друг между другом нескоррелированы.
        """
        correlation_matrix = features.corr(numeric_only=True)

        diagonal = np.eye(correlation_matrix.shape[0], dtype=bool)
        correlated_matrix = (
            correlation_matrix.mask(diagonal, False).abs() >= self.thresh
        )

        columns = self.filter(correlated_matrix)

        return features.loc[:, columns]
