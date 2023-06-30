import numpy as np
import pyfds.fields as fls
import pyfds.regions as reg


def test_dimension():
    dim = fls.Dimension(3, 0.1)
    assert np.allclose(dim.vector, np.asarray([0, 0.1, 0.2]))
    assert dim.get_index(0.1) == 1


def test_field_component_boundary_1():
    fc = fls.FieldComponent(100)
    fc.values = np.random.rand(100)
    fc.boundaries = [reg.Boundary(reg.LineRegion([5, 6, 7], [0, 0.2], 'test boundary'))]
    fc.boundaries[0].value = 23
    fc.apply_bounds(step=0)
    assert np.allclose(fc.values[[5, 6, 7]], [23, 23, 23])


def test_field_component_boundary_2():
    fc = fls.FieldComponent(100)
    fc.values = np.ones(100)
    fc.boundaries = [reg.Boundary(reg.LineRegion([5, 6, 7], [0, 0.2], 'test boundary'))]
    fc.boundaries[0].value = [np.arange(0, 3) * 23, np.arange(0, 3) * 42, np.arange(0, 3) * 23]
    fc.boundaries[0].additive = True
    fc.apply_bounds(step=1)
    assert np.allclose(fc.values[[5, 6, 7]], [24, 43, 24])


def test_field_component_boundary_3():
    fc = fls.FieldComponent(100)
    fc.values = np.ones(100)
    fc.boundaries = [reg.Boundary(reg.LineRegion([5, 6, 7], [0, 0.2], 'test boundary'))]
    fc.boundaries[0].value = np.arange(0, 3) * 23
    fc.boundaries[0].additive = True
    fc.apply_bounds(step=2)
    assert np.allclose(fc.values[[5, 6, 7]], [47, 47, 47])


def test_field_component_output():
    fc = fls.FieldComponent(100)
    fc.outputs = [reg.Output(reg.LineRegion([0, 1, 2], [0, 0.2], 'test output'))]
    fc.write_outputs()
    fc.write_outputs()
    assert np.allclose(fc.outputs[0].signals, [[0, 0], [0, 0], [0, 0]])
    assert np.allclose(fc.outputs[0].mean_signal, np.zeros(2))


def test_field1d_init():
    # create a field where the main material is 5
    fld = fls.Field1D(100, 0.1, 100, 0.1, int(5))
    # check if the "material parameter" 'real' for the complete field is 5
    assert np.allclose(fld.material_vector('real'), 5)


def test_field1d_d_x():
    fld = fls.Field1D(3, 1, 3, 1, int(5))
    assert np.allclose(fld.d_x().toarray(), [[-1, 1, 0], [0, -1, 1], [0, 0, -1]])
    assert np.allclose(fld.d_x(variant='backward').toarray(), [[1, 0, 0], [-1, 1, 0], [0, -1, 1]])
    assert np.allclose(fld.d_x(variant='central').toarray(), [[0, 0.5, 0], [-0.5, 0, 0.5],
                                                              [0, -0.5, 0]])


def test_field1d_d_x2():
    fld = fls.Field1D(3, 1, 3, 1, 5)
    assert np.allclose(fld.d_x2().toarray(), [[-2, 1, 0], [1, -2, 1], [0, 1, -2]])


def test_field2d_init():
    # create a field where the main material is 5
    fld = fls.Field2D(100, 0.1, 100, 0.1, 100, 0.1, int(5))
    # check if the "material parameter" 'real' for the complete field is 5
    assert np.allclose(fld.material_vector('real'), 5)
    assert np.size(fld.material_vector('real')) == 10000


def test_field2d_d_x():
    fld = fls.Field2D(2, 1, 2, 1, 10, 1, int(5))
    assert np.allclose(fld.d_x().toarray(), [[-1, 1, 0, 0], [0, -1, 1, 0],
                                             [0, 0, -1, 1], [0, 0, 0, -1]])
    assert np.allclose(fld.d_x(variant='backward').toarray(), [[1, 0, 0, 0], [-1, 1, 0, 0],
                                                               [0, -1, 1, 0], [0, 0, -1, 1]])
    assert np.allclose(fld.d_x(variant='central').toarray(), [[0, 0.5, 0, 0], [-0.5, 0, 0.5, 0],
                                                              [0, -0.5, 0, 0.5], [0, 0, -0.5, 0]])


def test_field2d_d_x2():
    fld = fls.Field2D(2, 1, 2, 1, 10, 1, int(5))
    assert np.allclose(fld.d_x2().toarray(), [[-2, 1, 0, 0], [1, -2, 1, 0],
                                              [0, 1, -2, 1], [0, 0, 1, -2]])


def test_field2d_d_y():
    fld = fls.Field2D(2, 1, 2, 1, 10, 1, int(5))
    assert np.allclose(fld.d_y().toarray(), [[-1, 0, 1, 0], [0, -1, 0, 1],
                                             [0, 0, -1, 0], [0, 0, 0, -1]])
    assert np.allclose(fld.d_y(variant='backward').toarray(), [[1, 0, 0, 0], [0, 1, 0, 0],
                                                               [-1, 0, 1, 0], [0, -1, 0, 1]])
    assert np.allclose(fld.d_y(variant='central').toarray(), [[0, 0, 0.5, 0], [0, 0, 0, 0.5],
                                                              [-0.5, 0, 0, 0], [0, -0.5, 0, 0]])


def test_field2d_d_y2():
    fld = fls.Field2D(2, 1, 2, 1, 10, 1, int(5))
    assert np.allclose(fld.d_y2().toarray(), [[-2, 0, 1, 0], [0, -2, 0, 1],
                                              [1, 0, -2, 0], [0, 1, 0, -2]])


def test_field2d_get_index():
    fld = fls.Field2D(4, 0.1, 3, 0.1, 1, 1, int(5))
    assert fld.get_index((0.2, 0.1)) == 6


def test_field1d_get_position():
    fld = fls.Field1D(4, 0.1, 1, 1, int(5))
    assert np.allclose(fld.get_position(fld.get_index(0.1)), 0.1)


def test_field2d_get_position():
    fld = fls.Field2D(4, 0.1, 3, 0.1, 1, 1, int(5))
    assert np.allclose(fld.get_position(fld.get_index((0.2, 0.1))), (0.2, 0.1))


def test_field1d_get_line_region():
    fld = fls.Field1D(4, 0.1, 1, 1, int(5))
    fld.material_regions.append(reg.MaterialRegion(fld.get_line_region((0.1, 0.2)), int(23)))
    assert np.allclose(fld.material_vector('real'), [5, 23, 23, 5])


def test_field2d_get_line_region():
    fld = fls.Field2D(3, 1, 4, 0.5, 1, 1, int(5))
    region = fld.get_line_region((1, 0, 1, 1.5))
    assert np.allclose(region.indices, [1, 4, 7, 10])
    region = fld.get_line_region((0, 0, 2, 0))
    assert np.allclose(region.indices, [0, 1, 2])
    region = fld.get_line_region((0, 0, 2, 1.5))
    assert np.allclose(region.indices, [0, 4, 7, 11])
    region = fld.get_line_region((0, 1.5, 2, 0))
    assert np.allclose(region.indices, [9, 7, 4, 2])


def test_field2d_get_rect_region():
    fld = fls.Field2D(3, 1, 4, 0.5, 1, 1, int(5))
    region = fld.get_rect_region((0, 0, 1, 1))
    assert np.allclose(region.indices, [0, 3, 6, 1, 4, 7])
    region = fld.get_rect_region((2, 1.5, -1, -1))
    assert np.allclose(region.indices, [4, 7, 10, 5, 8, 11])
