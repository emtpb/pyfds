import numpy as np


class Region:
    """Storage for the indices of points and metadata for which specific behaviour is to be
    applied (boundaries, materials, output, etc.)."""

    def __init__(self, indices, name=''):
        """Class constructor.

        Args:
            indices: Point indices of the region.
            name: Name of the region.
        """

        self.indices = indices
        self.name = name


class PointRegion(Region):
    """Region specified by individual points."""

    def __init__(self, indices, coordinates, name=''):
        """Class constructor.

        Args:
            indices: Point indices of the region.
            coordinates: Coordinates of the region.
            name: Name of the region.
        """

        super().__init__(indices, name)
        self.point_coordinates = coordinates


class LineRegion(Region):
    """Region specified by a line of points."""

    def __init__(self, indices, coordinates, name=''):
        """Class constructor.

        Args:
            indices: Point indices of the region.
            coordinates: Coordinates of the region.
            name: Name of the region.
        """

        super().__init__(indices, name)
        self.line_coordinates = coordinates


class RectRegion(Region):
    """Region specified by a rectangular field of points."""

    def __init__(self, indices, coordinates, name=''):
        """Class constructor.

        Args:
            indices: Point indices of the region.
            coordinates: Coordinates of the region.
            name: Name of the region.
        """

        super().__init__(indices, name)
        self.rect_coordinates = coordinates


class Boundary:
    """Specifies values that are to be written to a FieldComponent after each simulation step
    (like excitation signals and fixed boundaries)."""

    def __init__(self, region, value=0, additive=False):
        """Class constructor.

        Args:
            region: Region the boundary is applied to.
            value: Value the boundary applies to the field. May be scalar, list of scalars, signal
                as numpy array or list of signals.
            additive: Specifies if the boundary is additive to the field or if the fields value is
                set directly.
        """

        self.region = region
        self.value = value
        self.additive = additive


class Output:
    """Specifies values to be extracted from the FieldComponent after each simulation step."""

    def __init__(self, region):
        """Class constructor.

        Args:
            region: Region the out is recorded at.
        """

        self.region = region
        self.signals = []

    @property
    def mean_signal(self):
        """Return the mean signal of all points in the region."""

        return np.mean(np.asarray(self.signals), axis=0)


class MaterialRegion:
    """Specifies material(s) for a given region."""

    def __init__(self, region, material):
        """Class constructor.

        Args:
            region: Region the material is set.
            material: Material of the specified region.
        """

        self.region = region
        self.materials = [material]
