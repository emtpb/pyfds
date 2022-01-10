import logging as lo
import numpy as np
from . import fields as fld

logger = lo.getLogger('pyfds')


class Thermal1D(fld.Field1D):
    """Class for simulation of one-dimensional thermal fields."""

    def __init__(self, *args, **kwargs):
        """Class constructor.

        Args:
            See pyfds.fields.Field1D constructor arguments.
        """
        super().__init__(*args, **kwargs)
        self.temperature = fld.FieldComponent(self.num_points)
        self.heat_flux = fld.FieldComponent(self.num_points)

        # initialize attributes sparse matrices
        self.a_t_q = None
        self.a_q_t = None

    def assemble_matrices(self):
        """Assemble the a_* matrices required for simulation."""

        self.a_t_q = self.d_x(factors=(self.t.increment / self.x.increment /
                                       self.material_vector('density') /
                                       self.material_vector('heat_capacity')))
        self.a_q_t = self.d_x(factors=(1 / self.x.increment *
                                       self.material_vector('thermal_conductivity_x')),
                              variant='backward')
        self.matrices_assembled = True

    def sim_step(self):
        """Simulate one step."""

        self.temperature.apply_bounds(self.step)
        self.temperature.write_outputs()

        self.heat_flux.values = -self.a_q_t.dot(self.temperature.values)

        self.heat_flux.apply_bounds(self.step)
        self.heat_flux.write_outputs()

        self.temperature.values -= self.a_t_q.dot(self.heat_flux.values)


class Thermal2D(fld.Field2D):
    """Class for simulation of two-dimensional thermal fields."""

    def __init__(self, *args, **kwargs):
        """Class constructor.

        Args:
            See pyfds.fields.Field2D constructor arguments.
        """

        super().__init__(*args, **kwargs)
        self.temperature = fld.FieldComponent(self.num_points)
        self.heat_flux_x = fld.FieldComponent(self.num_points)
        self.heat_flux_y = fld.FieldComponent(self.num_points)

        # initialize attributes sparse matrices
        self.a_t_qx = None
        self.a_t_qy = None
        self.a_qx_t = None
        self.a_qy_t = None

    def assemble_matrices(self):
        """Assemble the a_* matrices required for simulation."""

        self.a_t_qx = self.d_x(factors=(self.t.increment / self.x.increment /
                                        self.material_vector('density') /
                                        self.material_vector('heat_capacity')))
        self.a_t_qy = self.d_y(factors=(self.t.increment / self.y.increment /
                                        self.material_vector('density') /
                                        self.material_vector('heat_capacity')))
        self.a_qx_t = self.d_x(factors=(1 / self.x.increment *
                                        self.material_vector('thermal_conductivity_x')),
                               variant='backward')
        self.a_qy_t = self.d_y(factors=(1 / self.y.increment *
                                        self.material_vector('thermal_conductivity_y')),
                               variant='backward')
        self.matrices_assembled = True

    def sim_step(self):
        """Simulate one step."""

        self.temperature.apply_bounds(self.step)
        self.temperature.write_outputs()

        self.heat_flux_x.values = -self.a_qx_t.dot(self.temperature.values)
        self.heat_flux_y.values = -self.a_qy_t.dot(self.temperature.values)

        self.heat_flux_x.apply_bounds(self.step)
        self.heat_flux_x.write_outputs()
        self.heat_flux_y.apply_bounds(self.step)
        self.heat_flux_y.write_outputs()

        self.temperature.values -= (self.a_t_qx.dot(self.heat_flux_x.values) +
                                    self.a_t_qy.dot(self.heat_flux_y.values))


class Thermal3DAxi(fld.Field2D):
    """Class for simulation of three-dimensional, axial-symmetric thermal fields. Note the x is
    the radial direction, and y is the z direction."""

    def __init__(self, *args, **kwargs):
        """Class constructor.

        Args:
            See pyfds.fields.Field2D constructor arguments.
        """

        super().__init__(*args, **kwargs)
        self.temperature = fld.FieldComponent(self.num_points)
        self.heat_flux_x = fld.FieldComponent(self.num_points)
        self.heat_flux_y = fld.FieldComponent(self.num_points)

        # initialize attributes sparse matrices
        self.a_t_qx = None
        self.a_t_qy = None
        self.a_qx_t = None
        self.a_qy_t = None

    def _radii(self):
        """Returns an array the same size as self.num_points with the distance from the y-axis
        (i.e. the x coordinate) of each heat flux point (hence x.increment/2 is added). For axial-
        symmetric fields, this is the radius, which is required to formulate the differential
        operators.

        Returns
            Radius of each heat flux point.
        """

        return np.tile(self.x.vector, self.y.samples) + self.x.increment/2

    def assemble_matrices(self):
        """Assemble the a_* matrices required for simulation."""

        self.a_t_qx = self.d_x(factors=(self.t.increment / self.x.increment /
                                        self.material_vector('density') /
                                        self.material_vector('heat_capacity') / self._radii()))
        self.a_t_qy = self.d_y(factors=(self.t.increment / self.y.increment /
                                        self.material_vector('density') /
                                        self.material_vector('heat_capacity')))
        self.a_qx_t = self.d_x(factors=(1 / self.x.increment *
                                        self.material_vector('thermal_conductivity_x')),
                               variant='backward')
        self.a_qy_t = self.d_y(factors=(1 / self.y.increment *
                                        self.material_vector('thermal_conductivity_y')),
                               variant='backward')
        self.matrices_assembled = True

    def sim_step(self):
        """Simulate one step."""

        self.temperature.apply_bounds(self.step)
        self.temperature.write_outputs()

        self.heat_flux_x.values = -self.a_qx_t.dot(self.temperature.values)
        self.heat_flux_y.values = -self.a_qy_t.dot(self.temperature.values)

        self.heat_flux_x.apply_bounds(self.step)
        self.heat_flux_x.write_outputs()
        self.heat_flux_y.apply_bounds(self.step)
        self.heat_flux_y.write_outputs()

        self.temperature.values -= (self.a_t_qx.dot(self.heat_flux_x.values * self._radii()) +
                                    self.a_t_qy.dot(self.heat_flux_y.values))


class ThermalMaterial:
    """Class for specification of thermal material parameters."""

    def __init__(self, heat_capacity, density, thermal_conductivity):
        """Class constructor.

        Args:
            heat_capacity: Specific heat capacity.
            density: Density.
            thermal_conductivity: Thermal conductivity coefficient.
        """

        self.heat_capacity = heat_capacity
        self.density = density
        self.thermal_conductivity_x = None
        self.thermal_conductivity_y = None
        self.thermal_conductivity = thermal_conductivity

    @property
    def thermal_conductivity(self):
        return self.thermal_conductivity_x, self.thermal_conductivity_y

    @thermal_conductivity.setter
    def thermal_conductivity(self, value):
        if isinstance(value, (list, tuple, np.ndarray)) and \
                len(value) == 2:
            self.thermal_conductivity_x = value[0]
            self.thermal_conductivity_y = value[1]
        elif isinstance(value, (float, int)):
            self.thermal_conductivity_x = value
            self.thermal_conductivity_y = value
        else:
            raise ValueError('Thermal conductivity must either be scalar or a 2 element vector.')
