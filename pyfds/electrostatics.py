import logging as lo
import scipy.sparse as sp
import scipy.sparse.linalg as sl
import warnings as wn
from . import fields as fld

logger = lo.getLogger('pyfds')


class Electrostatic1D(fld.Field1D):
    """Class for simulation of one-dimensional electrostatic fields."""

    def __init__(self, x_samples, x_delta, material):
        """Class constructor.

        Args:
            x_samples: Number of samples in x direction.
            x_delta: Increment in x direction.
            material: Main material of the field.
        """

        super().__init__(x_samples, x_delta, 1, 1, material)
        self.potential = fld.FieldComponent(self.num_points)
        self.charge_density = fld.FieldComponent(self.num_points)

        # initialize attributes sparse matrices
        self.a_phi_rho = None

    def assemble_matrices(self):
        """Assemble the a_* matrices required for simulation."""

        # Conversion to Compressed Sparse Column format is necessary for efficient inversion
        a_rho_phi = sp.csc_matrix(
            self.d_x2(factors=(self.material_vector('permittivity') /
                               self.x.increment ** 2)))
        self.a_phi_rho = sl.inv(a_rho_phi)
        self.matrices_assembled = True

    def sim_step(self):
        """Simulate one step."""

        self.charge_density.apply_bounds(self.step)
        self.charge_density.write_outputs()

        self.potential.values = -self.a_phi_rho.dot(self.charge_density.values)

        if len(self.potential.boundaries) != 0:
            wn.warn('Boundary conditions for electric potential are currently not supported.',
                    stacklevel=2)
            logger.warning(
                'Boundary conditions for electric potential are currently not supported.')


class Electrostatic2D(fld.Field2D):
    """Class for simulation of two-dimensional electrostatic fields."""

    def __init__(self, x_samples, x_delta, y_samples, y_delta, material):
        """Class constructor.

        Args:
            x_samples: Number of samples in x direction.
            x_delta: Increment in x direction.
            y_samples: Number of samples in y direction.
            y_delta: Increment in y direction.
            material: Main material of the field.
        """

        super().__init__(x_samples, x_delta, y_samples, y_delta, 1, 1, material)
        self.potential = fld.FieldComponent(self.num_points)
        self.charge_density = fld.FieldComponent(self.num_points)

        # initialize attributes sparse matrices
        self.a_phi_rho = None

    def assemble_matrices(self):
        """Assemble the a_* matrices required for simulation."""

        # Conversion to Compressed Sparse Column format is necessary for efficient inversion
        a_rho_phi = sp.csc_matrix(
            self.d_x2(factors=(self.material_vector('permittivity') /
                               self.x.increment ** 2)) +
            self.d_y2(factors=(self.material_vector('permittivity') /
                               self.y.increment ** 2)))
        self.a_phi_rho = sl.inv(a_rho_phi)
        self.matrices_assembled = True

    def sim_step(self):
        """Simulate one step."""

        self.charge_density.apply_bounds(self.step)
        self.charge_density.write_outputs()

        self.potential.values = -self.a_phi_rho.dot(self.charge_density.values)

        if len(self.potential.boundaries) != 0:
            wn.warn('Boundary conditions for electric potential are currently not supported.',
                    stacklevel=2)
            logger.warning(
                'Boundary conditions for electric potential are currently not supported.')


class ElectrostaticMaterial:
    """Class for specification of electrostatic material parameters."""

    def __init__(self, permittivity=8.8541878128e-12):
        """Class constructor. Default values create vacuum.

        Args:
            permittivity: Permittivity in As/Vm.
        """
        self.permittivity = permittivity
