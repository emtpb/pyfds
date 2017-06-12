import numpy as np
import scipy.sparse as sp
import warnings as wn
from . import regions as reg


class Field:
    """Base class for all fields."""

    def __init__(self):
        self.material_regions = []
        self.step = 0

    @property
    def num_points(self):
        """Returns number of points in the field."""
        raise NotImplementedError

    def get_index(self, position):
        """Returns the index of a point at the given position.

        Args:
            position: Position of the requested point.

        Returns:
            Index of the point.
        """
        raise NotImplementedError

    def material_vector(self, mat_parameter):
        """Get a vector that contains the specified material parameter for every point of the
        field.

        Args:
            mat_parameter: Material parameter of interest.

        Returns:
            Vector which contains the specified material parameter for each point in the field.
        """

        param_found = False
        mat_vector = np.zeros(self.num_points)

        for mat_reg in self.material_regions:
            for mat in mat_reg.materials:
                if hasattr(mat, mat_parameter):
                    mat_vector[mat_reg.region.indices] = getattr(mat, mat_parameter)
                    param_found = True

        if not param_found:
            wn.warn('Material parameter {} not found in set materials. Returning zeros.'
                    .format(mat_parameter), stacklevel=2)

        return mat_vector

    def get_point_region(self, position, name=''):
        """Creates a point region at the given position.

        Args:
            position: Position of the point region.
            name: Name of the point region.

        Returns:
            Point region.
        """

        return reg.PointRegion([self.get_index(position)], position, name=name)

    def add_material_region(self, *args, **kwargs):
        """Adds a material region to the field.

        Args:
            See pyfds.regions.MaterialRegion constructor arguments.
        """

        new_material_region = reg.MaterialRegion(*args, **kwargs)
        self.material_regions.append(new_material_region)


class Field1D(Field):
    """Class for one-dimensional fields."""

    def __init__(self, x_samples, x_delta, t_samples, t_delta, material):
        """Class constructor.

        Args:
            x_samples: Number of samples in x direction.
            x_delta: Increment in x direction.
            t_samples: Number of time samples.
            t_delta: Time increment.
            material: Main material of the field.
        """

        super().__init__()
        self.x = Dimension(x_samples, x_delta)
        self.t = Dimension(t_samples, t_delta)

        # add main material
        self.add_material_region(self.get_line_region((0, max(self.x.vector)), name='main'),
                                 material)

    @property
    def num_points(self):
        """Returns number of points in the field."""

        return self.x.samples

    def d_x(self, factors=None, variant='forward'):
        """Creates a sparse matrix for computing the first derivative with respect to x multiplied
        by factors given for every point. Uses forward difference quotient by default.

        Args:
            factors: Factor for each point to be applied after derivation.
            variant: Variant for the difference quotient ('forward', 'central', or 'backward').

        Returns:
            Sparse matrix the calculate derivatives of field components.
        """

        # use ones as factors if none are specified
        if factors is None:
            factors = np.array(1).repeat(self.num_points)

        if variant == 'forward':
            return sp.dia_matrix((np.array([-factors, factors]), [0, 1]),
                                 shape=(self.num_points, self.num_points))
        elif variant == 'central':
            return sp.dia_matrix((np.array([-factors / 2, factors / 2]), [-1, 1]),
                                 shape=(self.num_points, self.num_points))
        elif variant == 'backward':
            return sp.dia_matrix((np.array([-factors, factors]), [-1, 0]),
                                 shape=(self.num_points, self.num_points))
        else:
            raise ValueError('Unknown difference quotient variant {}.'.format(variant))

    def d_x2(self, factors=None):
        """Creates a sparse matrix for computing the second derivative with respect to x multiplied
        by factors given for every point. Uses central difference quotient.

        Args:
            factors: Factor for each point to be applied after derivation.

        Returns:
            Sparse matrix the calculate second derivatives of field components.
        """

        # use ones as factors if none are specified
        if factors is None:
            factors = np.array(1).repeat(self.num_points)

        return sp.dia_matrix((np.array([factors, -2*factors, factors]), [-1, 0, 1]),
                             shape=(self.num_points, self.num_points))

    def get_index(self, position):
        """Returns the index of a point at the given position.

        Args:
            position: Position of the requested point.

        Returns:
            Index of the point.
        """

        return self.x.get_index(position)

    def get_position(self, index):
        """Returns the position of a point with the given index.

        Args:
            index: Index of a point.

        Returns:
            Position of the point as x coordinate.
        """

        return self.x.vector[index]

    def get_line_region(self, position, name=''):
        """Creates a line region at the given position (start, end), inclusive.

        Args:
            position: Position of the line region (start, end), as x coordinates.
            name: Name of the region.

        Returns:
            Line region.
        """

        return reg.LineRegion([index for index in range(self.get_index(position[0]),
                                                        self.get_index(position[1]) + 1)],
                              position, name=name)


class Field2D(Field):
    """Class for two-dimensional fields."""

    def __init__(self, x_samples, x_delta, y_samples, y_delta, t_samples, t_delta, material):
        """Class constructor.

        Args:
            x_samples: Number of samples in x direction.
            x_delta: Increment in x direction.
            y_samples: Number of samples in y direction.
            y_delta: Increment in y direction.
            t_samples: Number of time samples.
            t_delta: Time increment.
            material: Main material of the field.
        """

        super().__init__()
        self.x = Dimension(x_samples, x_delta)
        self.y = Dimension(y_samples, y_delta)
        self.t = Dimension(t_samples, t_delta)

        # add main material
        self.add_material_region(self.get_rect_region(
            (0, 0, max(self.x.vector), max(self.y.vector)), name='main'), material)

    @property
    def num_points(self):
        return self.x.samples * self.y.samples

    def d_x(self, factors=None, variant='forward'):
        """Creates a sparse matrix for computing the first derivative with respect to x multiplied
        by factors given for every point. Uses forward difference quotient by default.

        Args:
            factors: Factor for each point to be applied after derivation.
            variant: Variant for the difference quotient ('forward', 'central', or 'backward').

        Returns:
            Sparse matrix the calculate derivatives of field components.
        """

        # use ones as factors if none are specified
        if factors is None:
            factors = np.array(1).repeat(self.num_points)

        if variant == 'forward':
            return sp.dia_matrix((np.array([-factors, factors]), [0, 1]),
                                 shape=(self.num_points, self.num_points))
        elif variant == 'central':
            return sp.dia_matrix((np.array([-factors/2, factors/2]), [-1, 1]),
                                 shape=(self.num_points, self.num_points))
        elif variant == 'backward':
            return sp.dia_matrix((np.array([-factors, factors]), [-1, 0]),
                                 shape=(self.num_points, self.num_points))
        else:
            raise ValueError('Unknown difference quotient variant {}.'.format(variant))

    def d_y(self, factors=None, variant='forward'):
        """Creates a sparse matrix for computing the first derivative with respect to y multiplied
        by factors given for every point. Uses forward difference quotient by default.

        Args:
            factors: Factor for each point to be applied after derivation.
            variant: Variant for the difference quotient ('forward', 'central', or 'backward').

        Returns:
            Sparse matrix the calculate derivatives of field components.
        """

        # use ones as factors if none are specified
        if factors is None:
            factors = np.array(1).repeat(self.num_points)

        if variant == 'forward':
            return sp.dia_matrix((np.array([-factors, factors]), [0, self.x.samples]),
                                 shape=(self.num_points, self.num_points))
        elif variant == 'central':
            return sp.dia_matrix(
                (np.array([-factors/2, factors/2]), [-self.x.samples, self.x.samples]),
                shape=(self.num_points, self.num_points))
        elif variant == 'backward':
            return sp.dia_matrix((np.array([-factors, factors]), [-self.x.samples, 0]),
                                 shape=(self.num_points, self.num_points))
        else:
            raise ValueError('Unknown difference quotient variant {}.'.format(variant))

    def d_x2(self, factors=None):
        """Creates a sparse matrix for computing the second derivative with respect to x multiplied
        by factors given for every point. Uses central difference quotient.

        Args:
            factors: Factor for each point to be applied after derivation.

        Returns:
            Sparse matrix the calculate second derivatives of field components.
        """

        # use ones as factors if none are specified
        if factors is None:
            factors = np.array(1).repeat(self.num_points)

        return sp.dia_matrix((np.array([factors, -2*factors, factors]), [-1, 0, 1]),
                             shape=(self.num_points, self.num_points))

    def d_y2(self, factors=None):
        """Creates a sparse matrix for computing the second derivative with respect to y multiplied
        by factors given for every point. Uses central difference quotient.

        Args:
            factors: Factor for each point to be applied after derivation.

        Returns:
            Sparse matrix the calculate second derivatives of field components.
        """

        # use ones as factors if none are specified
        if factors is None:
            factors = np.array(1).repeat(self.num_points)

        return sp.dia_matrix((np.array([factors, -2*factors, factors]),
                              [-self.x.samples, 0, self.x.samples]),
                             shape=(self.num_points, self.num_points))

    def get_index(self, position):
        """Returns the index of a point at the given position.

        Args:
            position: Position of the requested point.

        Returns:
            Index of the point.
        """

        return self.x.get_index(position[0]) + self.y.get_index(position[1]) * self.x.samples

    def get_position(self, index):
        """Returns the position of a point with the given index.

        Args:
            index: Index of a point.

        Returns:
            Position of the point as (x coordinate, y coordinate).
        """

        return self.x.vector[index % self.x.samples], self.y.vector[int(index / self.x.samples)]

    def get_line_region(self, position, name=''):
        """Creates a line region at the given position (start_x, start_y, end_x, end_y),
        inclusive.

        Args:
            position: Position of the line region (start_x, start_y, end_x, end_y).
            name: Name of the region.

        Returns:
            Line region.
        """

        start_idx = self.get_index(position[:2])
        end_idx = self.get_index(position[2:])

        x_diff = start_idx % self.x.samples - end_idx % self.x.samples
        y_diff = int(start_idx / self.x.samples) - int(end_idx / self.x.samples)

        num_points = max(np.abs([x_diff, y_diff]))
        point_indices = []

        for ii in range(num_points + 1):

            x_position = start_idx % self.x.samples - np.round(ii / num_points * x_diff)
            y_position = int(start_idx / self.x.samples) - np.round(ii / num_points * y_diff)
            point_indices.append(int(x_position + self.x.samples * y_position))

        return reg.LineRegion(point_indices, position, name=name)

    def get_rect_region(self, position, name=''):
        """Creates a rectangular region at the given position (origin_x, origin_y, size_x, size_y),
        inclusive, origin is the lower left corner.

        Args:
            position: Position of the rectangular region origin_x, origin_y, size_x, size_y).
            name: Name of the region.

        Returns:
            Rectangular region.
        """

        x_start = self.x.get_index(position[0])
        y_start = self.y.get_index(position[1])
        x_end = self.x.get_index(position[0] + position[2])
        y_end = self.y.get_index(position[1] + position[3])

        x_start, x_end = min(x_start, x_end), max(x_start, x_end)
        y_start, y_end = min(y_start, y_end), max(y_start, y_end)

        return reg.RectRegion([x + y * self.x.samples for x in range(x_start, x_end + 1)
                               for y in range(y_start, y_end + 1)], position, name)


class Dimension:
    """Represents a space or time axis."""

    def __init__(self, samples, increment):
        """Class constructor.

        Args:
            samples: Number of samples in the axis.
            increment: Increment between samples.
        """

        self.samples = int(samples)
        self.increment = increment
        self.snap_radius = np.finfo(float).eps * 10

    @property
    def vector(self):
        """Returns the axis as a vector."""

        return np.arange(start=0, stop=self.samples) * self.increment

    def get_index(self, value):
        """Returns the index of a given value.

        Args:
            value: Value the index requested for.

        Returns:
            Index.
        """

        index, = np.where(np.abs(self.vector - value) <= self.snap_radius)
        assert len(index) < 2, "Multiple points found within snap radius of given value."
        assert len(index) > 0, "No point found within snap radius of given value."

        return int(index)


class FieldComponent:
    """A single component of a field (e.g. electric field in the x direction)."""

    def __init__(self, num_points):
        """Class constructor.

        Args:
            num_points: Number of points in the field component.
        """

        # values of the field component
        self.values = np.zeros(num_points)
        # list with objects of type Boundary
        self.boundaries = []
        # list with objects of type Output
        self.outputs = []

    def apply_bounds(self, step):
        """Applies the boundary conditions to the field component.

        Args:
            step: Simulation step, required if boundary is a signal that changes of time.
        """

        for bound in self.boundaries:
            self.values[bound.region.indices] = bound.apply(self.values[bound.region.indices],
                                                            step=step)

    def write_outputs(self):
        """Writes the values of the field component to the outputs."""

        for output in self.outputs:

            if not output.signals:
                output.signals = [[self.values[index]] for index in output.region.indices]
            else:
                [signal.append(self.values[index]) for index, signal in
                 zip(output.region.indices, output.signals)]

    def add_boundary(self, *args, **kwargs):
        """Adds a boundary to the field component.

        Args:
            See pyfds.regions.Boundary constructor arguments.
        """

        new_bound = reg.Boundary(*args, **kwargs)
        self.boundaries.append(new_bound)

    def add_output(self, *args, **kwargs):
        """Adds output to the field component.

        Args:
            See pyfds.regions.Output constructor arguments.
        """

        new_output = reg.Output(*args, **kwargs)
        self.outputs.append(new_output)
