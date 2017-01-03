import numpy as np
import pyfds as fds


def test_acoustic_material():
    water = fds.AcousticMaterial(1500, 1000)
    water.bulk_viscosity = 1e-3
    water.shear_viscosity = 1e-3
    assert np.isclose(water.absorption_coef, 7e-3 / 3)
