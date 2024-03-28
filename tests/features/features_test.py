import numpy as np
import pandas as pd

df = pd.DataFrame(
    {
        "A": (np.exp2(range(7, 10)) - 1).astype(int),
        "B": np.sin(np.arange(7, 10)),
        "C": np.cos(np.arange(7, 10)),
        "D": ["foo", None, "baz"],
        "E": ["who", "what", "when"],
    }
)
