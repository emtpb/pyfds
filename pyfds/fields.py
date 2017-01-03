import numpy as np
import scipy.sparse as sp
from . import regions as reg


class Field:
    """Base class for all fields."""

    def material_vector(self, mat_parameter):
        """Get a vector that contains the specified material parameter for every point of the
        field."""

        mat_vector = np.zeros(self.num_points)

        for mat_reg in self.material_regions:
            for mat in mat_reg.materials:
                if hasattr(mat, mat_parameter):
                    mat_vector[mat_reg.region.indices] = getattr(mat, mat_parameter)

        return mat_vector


class Field1D(Field):
    """Class for one dimensional fields."""

    def __init__(self, x_samples, x_delta, t_samples, t_delta, material):
        self.x = Dimension(x_samples, x_delta)
        self.t = Dimension(t_samples, t_delta)

        # add main material
        self.material_regions = [reg.MaterialRegion(reg.LineRegion(
            np.arange(self.x.samples, dtype='int_'), [0, max(self.x.vector)], 'main'), material)]

    @property
    def num_points(self):
        return self.x.samples

    def d_x(self, backward=False):
        """Creates a sparse matrix for computing the first derivative with respect to x.
        Uses forward difference quotient by default, specify backward=True if required otherwise"""

        if not backward:
            return sp.dia_matrix((np.array([[-1], [1]]).repeat(self.num_points, axis=1) /
                                  self.x.increment, [0, 1]),
                                 shape=(self.num_points, self.num_points))
        else:
            return sp.dia_matrix((np.array([[-1], [1]]).repeat(self.num_points, axis=1) /
                                  self.x.increment, [-1, 0]),
                                 shape=(self.num_points, self.num_points))


class Dimension:
    """Represents a space or time axis."""

    def __init__(self, samples, increment):

        self.samples = int(samples)
        self.increment = increment
        self.snap_radius = np.finfo(float).eps * 10

    @property
    def vector(self):
        return np.arange(start=0, stop=self.samples) * self.increment

    def get_index(self, value):
        """Returns the index of a given value."""

        index, = np.where(np.abs(self.vector - value) <= self.snap_radius)
        assert len(index) < 2, "Multiple points found within snap radius of given value."
        assert len(index) > 0, "No point found within snap radius of given value."

        return int(index)


class FieldComponent:
    """A single component of a field (e.g. electric field in the x direction)."""

    def __init__(self, num_points):

        # values of the field component
        self.values = np.zeros(num_points)
        # list with objects of type Boundary
        self.boundaries = []
        # list with objects of type Output
        self.outputs = []

    def apply_bounds(self):
        """Applies the  boundary conditions to the field component."""

        for bound in self.boundaries:
            self.values[bound.region.indices] = (bound.additive *
                                                 self.values[bound.region.indices] + bound.value)

    def write_outputs(self):
        """Writes the values of the field component to the outputs."""

        for output in self.outputs:

            if not output.signals:
                output.signals = [[self.values[index]] for index in output.region.indices]
            else:
                [signal.append(self.values[index]) for index, signal in
                 zip(output.region.indices, output.signals)]

