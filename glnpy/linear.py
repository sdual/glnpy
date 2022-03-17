import numpy as np

from scipy.spatial import logit


class Linear:

    def __init__(self, size: int, input_size: int, context_size: int,
                 context_map_size: int, num_classes: int,
                 learning_rate: float, pred_clipping: float, weight_clipping: float,
                 bias: bool, context_bias: bool):
        self._num_classes: int = num_classes
        self._learning_rate: float = learning_rate
        self._pred_clipping: float = pred_clipping
        self._weight_clipping: float = weight_clipping

        self._bias: np.ndarray = np.random.uniform(
            low=logit(self._pred_clipping), high=logit(1.0 - self._pred_clipping),
            size=(1, 1, self._num_classes)
        )

        self._size = size - 1

        self._context_maps: np.ndarray = np.random.normal(
            size=(self._num_classes, self._size,
                  context_map_size, context_size)
        )

        self._context_bias = np.random.normal(
            size=(self._num_classes, self._size, context_map_size, 1)
        )

        self._context_maps /= np.linalg.norm(self._context_map)

        self._boolean_converter = np.array(
            [[2**i] for i in range(context_map_size)])

        self._weights = np.full(
            shape=(self._num_classes, self._size, 2**context_map_size, input_size))

    def predict(self, logit, context, target=None):
        distances = np.matmul(self._context_maps, context.T)
        mapped_context_binary = (distances > self._context_bias).astype(int)
        pass
