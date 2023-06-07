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


class TriRegion(Region):
    """Region specified by a triangular field of points."""

    def __init__(self, indices, coordinates, name=''):
        """Class constructor.

        Args:
            indices: Point indices of the region.
            coordinates: Coordinates of the region.
            name: Name of the region.
        """

        super().__init__(indices, name)
        self.tri_coordinates = coordinates


class EllipseRegion(Region):
    """Region specified by a elliptic field of points."""

    def __init__(self, indices, centre, radii, name=''):
        """Class constructor.

        Args:
            indices: Point indices of the region.
            centre: Coordinates of the centre of the ellipse.
            radii: Radii/half axes of the ellipse in x and y direction.
            name: Name of the region.
        """

        super().__init__(indices, name)
        self.centre = centre
        self.radii = radii


class Boundary:
    """Specifies values that are to be written to a FieldComponent after each simulation step
    (like excitation signals and fixed boundaries)."""

    def __init__(self, region, value=0, additive=False):
        """Class constructor.

        Args:
            region: Region the boundary is applied to.
            value: Value the boundary applies to the field. May be scalar, signal as numpy array
                or list of signals.
            additive: Specifies if the boundary is additive to the field or if the fields value is
                set directly.
        """

        self.region = region
        self.value = value
        self.additive = additive

    def apply(self, old_values, step):
        """Apply the boundary.

        Args:
            old_values: Old values of the points in the boundary.
            step: Time step of the simulation (required if signals are to be applied).

        Returns:
            New values for the points in the boundary.
        """

        if np.ndim(self.value) == 0:
            # if a single value is given
                return self.additive * old_values + self.value
        elif type(self.value) == np.ndarray:
            # if a signal is given
            return self.additive * old_values + self.value[step]
        else:
            # if a list of signals for each index is given
            return [self.additive * old_values[ii] + signal[step]
                    for ii, signal in enumerate(self.value)]


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
