import numpy as np
import pyfds as fds


def test_acoustic_material():
    water = fds.AcousticMaterial(1500, 1000)
    water.bulk_viscosity = 1e-3
    water.shear_viscosity = 1e-3
    assert np.isclose(water.absorption_coef, 7e-3 / 3)
    water.absorption_coef = 3e-3
    assert np.isclose(water.absorption_coef, 3e-3)


def test_acoustic_material_absorption_coef():
    water = fds.AcousticMaterial(1500, 1000, absorption_coef=2e-3)
    assert np.isclose(water.absorption_coef, 2e-3)


def test_acoustic1d_create_matrices():
    fld = fds.Acoustic1D(t_delta=1, t_samples=1,
                         x_delta=1, x_samples=3,
                         material=fds.AcousticMaterial(700, 0.01, bulk_viscosity=1))
    fld.assemble_matrices()
    assert np.allclose(fld.a_p_v.toarray(), [[-4900, 4900, 0], [0, -4900, 4900], [0, 0, -4900]])
    assert np.allclose(fld.a_v_p.toarray(), [[100, 0, 0], [-100, 100, 0], [0, -100, 100]])
    assert np.allclose(fld.a_v_v.toarray(), [[-200, 100, 0], [100, -200, 100], [0, 100, -200]])


def test_acoustic2d_create_matrices():
    fld = fds.Acoustic2D(t_delta=1, t_samples=1,
                         x_delta=1, x_samples=2,
                         y_delta=1, y_samples=2,
                         material=fds.AcousticMaterial(700, 0.01, bulk_viscosity=1))
    fld.assemble_matrices()
    assert np.allclose(fld.a_p_vx.toarray(), [[-4900, 4900, 0, 0], [0, -4900, 4900, 0],
                                              [0, 0, -4900, 4900], [0, 0, 0, -4900]])
    assert np.allclose(fld.a_p_vy.toarray(), [[-4900, 0, 4900, 0], [0, -4900, 0, 4900],
                                              [0, 0, -4900, 0], [0, 0, 0, -4900]])
    assert np.allclose(fld.a_vx_p.toarray(), [[100, 0, 0, 0], [-100, 100, 0, 0], [0, -100, 100, 0],
                                              [0, 0, -100, 100]])
    assert np.allclose(fld.a_vy_p.toarray(), [[100, 0, 0, 0], [0, 100, 0, 0], [-100, 0, 100, 0],
                                              [0, -100, 0, 100]])
    assert np.allclose(fld.a_vx_vx.toarray(), [[-400, 100, 100, 0], [100, -400, 100, 100],
                                               [100, 100, -400, 100], [0, 100, 100, -400]])
    assert np.allclose(fld.a_vy_vy.toarray(), [[-400, 100, 100, 0], [100, -400, 100, 100],
                                               [100, 100, -400, 100], [0, 100, 100, -400]])


def test_acoustic3d_axi_create_matrices():
    fld = fds.Acoustic3DAxi(t_delta=1, t_samples=1,
                            x_delta=1, x_samples=2,
                            y_delta=1, y_samples=2,
                            material=fds.AcousticMaterial(1, 1, bulk_viscosity=1))
    fld.assemble_matrices()
    assert np.allclose(fld.a_p_vx.toarray(), [[-2, 2 / 3, 0, 0], [0, -2 / 3, 2, 0],
                                              [0, 0, -2, 2 / 3], [0, 0, 0, -2 / 3]])
    assert np.allclose(fld.a_p_vy.toarray(), [[-1, 0, 1, 0], [0, -1, 0, 1],
                                              [0, 0, -1, 0], [0, 0, 0, -1]])
    assert np.allclose(fld.a_vx_p.toarray(), [[1, 0, 0, 0], [-1, 1, 0, 0], [0, -1, 1, 0],
                                              [0, 0, -1, 1]])
    assert np.allclose(fld.a_vy_p.toarray(), [[1, 0, 0, 0], [0, 1, 0, 0], [-1, 0, 1, 0],
                                              [0, -1, 0, 1]])
    assert np.allclose(fld.a_vx_vx.toarray(), [[-4, 4 / 3, 1, 0], [0, -4, 2, 1],
                                               [1, 2 / 3, -4, 4 / 3], [0, 1, 0, -4]])
    assert np.allclose(fld.a_vy_vy.toarray(), [[-4, 4 / 3, 1, 0], [0, -4, 2, 1],
                                               [1, 2 / 3, -4, 4 / 3], [0, 1, 0, -4]])
