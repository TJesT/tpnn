import numpy as np

# data and targets presents in form
# (DATA_SAMPLE_SIZE, DIM1, DIM2)

type TrainData = np.ndarray[float]
type TrainTargets = np.ndarray[float]
type TestData = np.ndarray[float]
type TestTargets = np.ndarray[float]


def train_test_split(
    data: np.ndarray[float],
    targets: np.ndarray[float],
    *,
    target_portion: float = 0.2,
    data_portion: float = 0.8,
    shuffle: bool = True,
    random_state: int = None,
) -> tuple[TrainData, TrainTargets, TestData, TestTargets]:
    """
    NOTE: If target_portion and data_portion don't sum up to 1,
          then they will be rescaled linearly so that they summed up tp 1.
    """
    if random_state is not None:
        np.random.seed(random_state)

    data_portion = data_portion / (target_portion + data_portion)

    train_size = int(data_portion * data.shape[0])

    permutation = np.random.permutation(data.shape[0])
    permuted_data, permuted_targets = data[permutation], targets[permutation]

    return (
        permuted_data[:train_size],
        permuted_targets[:train_size],
        permuted_data[train_size:],
        permuted_targets[train_size:],
    )
