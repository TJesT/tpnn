import numpy as np
import pandas as pd
from tpnn.features import (
    rsr_any,
    rsr_except_one,
    centerize,
    normalize,
    replace_categorical_values,
    remove_outliers_zscore,
    remove_na,
)

df = pd.DataFrame(
    {
        "A": range(7, 10),
        "B": np.sin(np.arange(7, 10)),
        "C": np.cos(np.arange(7, 10)),
        "D": ["foo", "bar", "baz"],
        "E": ["who", "what", "when"],
    }
)

print(df)
print(rsr_any(df))
print(rsr_except_one(df))

print(centerize(df))
print(normalize(df))
print(replace_categorical_values(df))
print(remove_outliers_zscore(df))
remove_na(df)
# rsr_except_one(df, inplace=True)
remove_outliers_zscore(df, inplace=True)
print(df)
