import numpy as np


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

