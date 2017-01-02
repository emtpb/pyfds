import numpy as np


class Region:
    """Storage for the indices of points and metadata for which specific behaviour is to be
    applied (boundaries, materials, output, etc.)."""

    def __init__(self, indices, name=''):

        # point indices given as array
        self.indices = indices
        # name of the region (for convenience)
        self.name = name


class PointRegion(Region):
    """Region specified by individual points."""

    def __init__(self, indices, coordinates, name=''):

        super().__init__(indices, name)
        self.point_coordinates = coordinates


class LineRegion(Region):
    """Region specified by a line of points."""

    def __init__(self, indices, coordinates, name=''):

        super().__init__(indices, name)
        self.line_coordinates = coordinates


class RectRegion(Region):
    """Region specified by a rectangular field of points."""

    def __init__(self, indices, coordinates, name=''):

        super().__init__(indices, name)
        self.rect_coordinates = coordinates


class Boundary:
    """Specifies values that are to be written to a FieldComponent after each simulation step
    (like excitation signals and fixed boundaries)."""

    def __init__(self, region, value=0, additive=False):

        self.region = region
        self.value = value
        self.additive = additive


class Output:
    """Specifies values to be extracted from the FieldComponent after each simulation step."""

    def __init__(self, region):

        self.region = region
        self.signals = []

    @property
    def mean_signal(self):
        return np.mean(np.asarray(self.signals), axis=0)


class MaterialRegion:
    """Specifies material(s) for a given region."""

    def __init__(self, region, material):

        self.region = region
        self.materials = [material]
