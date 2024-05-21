# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.axes import Axes
# from typing import Callable

# from tpnn.architectures.neural_network import NeuralNetwork

# matplotlib.rcParams["animation.embed_limit"] = 2**128


# def animate(n: int):
#     nonlocal ax
#     ax: Axes
#     check_points = np.linspace(0, 2 * np.pi, num=1000)
#     check_points = check_points.reshape((1000, 1, 1))
#     sns.lineplot(
#         x=check_points.ravel(),
#         y=(check_points >> rperceptron).ravel(),
#         ax=ax,
#         label="test_predict",
#     )
#     sns.lineplot(
#         x=check_points.ravel(), y=np.sin(check_points.ravel()), ax=ax, label="test"
#     )
#     sns.lineplot(
#         x=data.ravel(), y=(data >> rperceptron).ravel(), ax=ax, label="train_predict"
#     )
#     return ax


# def create_animation_during_learning(
#     network: NeuralNetwork, animate: Callable[[int], Axes], epochs: int
# ):
#     fig, ax = plt.subplots()

#     animation = FuncAnimation(..., animate, frames=range(epochs))
