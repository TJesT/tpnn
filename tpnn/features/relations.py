import pandas as pd
import numpy as np

from typing import Literal, override
from ..core.pipeline import Pipeable
from ..core.types import Label, pdPipeable


def _filter_if_any(corr: pd.DataFrame) -> "pd.Series[Label]":
    """
    Получить некоррелируюшие признаки за счёт удаления всех, что
    скоррелированы хотя бы с одним другим признаком.
    """
    include = ~corr.all(axis=1)
    return include[include].index


def _filter_sequentially(corr: pd.DataFrame) -> "pd.Series[Label]":
    """
    Получить некоррелирующие признаки за счёт последовательного удаления
    наиболее скоррелированных.

    1) Выбрать строку с наибольшей суммой (если таких несколько, то первую).
    2) Удалить строку и столбец с найденным индексом.
    3) Повторять предыдущие два шага, пока сумма по таблице не будет равна 0.
    """

    def column_sum(row: "pd.Index[Label]") -> int:
        return corr[row].sum()

    while corr.values.sum() > 0:
        column: Label = max(corr.index, key=column_sum)
        corr.drop(columns=column, index=column, inplace=True)

    return corr.index


class CorelationFilter(pdPipeable):
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
            -`'seq'`: Последовательно удалять признак влияющий на остальные больше всего.

    """

    __strategy_mapping = {
        "all": _filter_if_any,
        "seq": _filter_sequentially,
    }

    def __init__(
        self,
        strategy: Literal["all", "seq"] = "all",
        *,
        thresh: float = 0.5,
    ) -> None:
        self.filter = self.__strategy_mapping.get(strategy)
        self.thresh = thresh

    @override
    def __call__(self, _input: pd.DataFrame) -> pd.DataFrame:
        """
        Убрать из датафрейма `features` сильно скоррелированные колонки.

        1) Построить матрицу корреляций для числовых фичей.
        2) Убрать диагональные единицы.
        3) Применить фильтр для выбора колонок.
        4) Вернуть часть `features` с нужными колонками.

        Args:
            `features`: pd.DataFrame
                Фрейм с признаками. Могут быть и категориальными, и числовыми.

        Returns:
            pd.DataFrame только c теми колонками, которые друг между другом нескоррелированы.
        """
        correlation_matrix = _input.corr(numeric_only=True)

        diagonal = np.eye(correlation_matrix.shape[0], dtype=bool)
        correlated_matrix = (
            correlation_matrix.mask(diagonal, False).abs() >= self.thresh
        )

        columns = self.filter(correlated_matrix)

        return _input.loc[:, columns]
