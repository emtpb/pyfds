import numpy as np
import pyfds as fds


def test_dimension():
    dim = fds.Dimension(3, 0.1)
    assert np.allclose(dim.vector, np.asarray([0, 0.1, 0.2]))
    assert dim.get_index(0.1) == 1
