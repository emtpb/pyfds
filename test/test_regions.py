import numpy as np
import pyfds.regions as reg


def test_output():
    out = reg.Output(reg.LineRegion([0, 1, 2], [0, 0.2], 'test output'))
    out.signals = [np.linspace(0, 1) for _ in range(len(out.region.indices))]
    assert np.allclose(out.mean_signal, np.linspace(0, 1))
