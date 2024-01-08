import numpy as np
import pyfds as fds
import pytest as pt


def test_synchronized_fields():
    acs = fds.Acoustic1D(t_delta=1, t_samples=1,
                         x_delta=1, x_samples=3,
                         material=fds.AcousticMaterial(400, 1))
    ths = fds.Thermal1D(t_delta=1, t_samples=1,
                        x_delta=1, x_samples=3,
                        material=fds.ThermalMaterial(1, 1, 1))
    with pt.raises(RuntimeError):
        fds.SynchronizedFields([acs, acs], [])

    cpl = fds.SynchronizedFields([acs, ths], [])
    assert(acs.velocity in vars(cpl).values())
    assert(acs.pressure in vars(cpl).values())
    assert(ths.temperature in vars(cpl).values())
    assert(ths.heat_flux in vars(cpl).values())
    assert(len(cpl.material_regions) == 2)
    cpl.assemble_matrices()
    assert(acs.matrices_assembled and ths.matrices_assembled)
    cpl.simulate(1)
    assert(acs.step == 1)
    assert(ths.step == 1)
    assert(cpl.step == 1)


def test_boundary_coupling():
    comp1 = fds.fields.FieldComponent(num_points=12)
    comp1.values = 2 * np.ones(12)
    comp2 = fds.fields.FieldComponent(num_points=12)
    comp2.values = np.ones(12)
    quad_coupling = fds.BoundaryCoupling(comp1, comp2, lambda x: x ** 2)
    quad_coupling.apply(0)
    assert(np.allclose(comp2.values, 5 * np.ones(12)))

    accu_coupling = fds.BoundaryCoupling(comp1, comp2, lambda x: x, False, True, 3)
    accu_coupling.apply(1)
    accu_coupling.apply(2)
    accu_coupling.apply(3)
    accu_coupling.apply(4)
    assert(np.allclose(comp2.values, 6 * np.ones(12)))
