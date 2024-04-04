import pandas as pd
from .pipeline import Pipeable

pdPipeable = Pipeable[pd.DataFrame, pd.DataFrame]
type Label = str
type Probabilty = float
