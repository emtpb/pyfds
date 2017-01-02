import numpy as np
import pyfds as fds


def test_dimension():
    dim = fds.Dimension(3, 0.1)
    assert np.allclose(dim.vector, np.asarray([0, 0.1, 0.2]))
    assert dim.get_index(0.1) == 1


def test_field_component_boundary_1():
    fc = fds.FieldComponent(100)
    fc.values = np.random.rand(100)
    fc.boundaries = [fds.Boundary(fds.LineRegion([5, 6, 7], [0, 0.2], 'test boundary'))]
    fc.boundaries[0].value = 23
    fc.apply_bounds()
    assert np.allclose(fc.values[[5, 6, 7]], [23, 23, 23])


def test_field_component_boundary_2():
    fc = fds.FieldComponent(100)
    fc.values = np.ones(100)
    fc.boundaries = [fds.Boundary(fds.LineRegion([5, 6, 7], [0, 0.2], 'test boundary'))]
    fc.boundaries[0].value = [23, 42, 23]
    fc.boundaries[0].additive = True
    fc.apply_bounds()
    assert np.allclose(fc.values[[5, 6, 7]], [24, 43, 24])


def test_field_component_output():
    fc = fds.FieldComponent(100)
    fc.outputs = [fds.Output(fds.LineRegion([0, 1, 2], [0, 0.2], 'test output'))]
    fc.write_outputs()
    fc.write_outputs()
    assert np.allclose(fc.outputs[0].signals, [[0, 0], [0, 0], [0, 0]])
    assert np.allclose(fc.outputs[0].mean_signal, np.zeros(2))


def test_field1d_init():
    # create a field where the main material is 5
    fld = fds.Field1D(100, 0.1, 100, 0.1, int(5))
    # check if the "material parameter" 'real' for the complete field is 5
    assert np.allclose(fld.material_vector('real'), 5)
