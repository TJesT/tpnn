import pandas as pd
import numpy as np
from .pipeline import Pipeable

pdPipeable = Pipeable[pd.DataFrame, pd.DataFrame]
type Label = str
type Probabilty = float

type BiasesGradient = np.ndarray[float]
type WeightsGradient = np.ndarray[float]
type BiasesGradients = list[BiasesGradient]
type WeightsGradients = list[WeightsGradient]
