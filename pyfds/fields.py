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
