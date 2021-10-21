from . import fields as fld


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
                                       self.material_vector('thermal_conductivity')),
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
                                        self.material_vector('thermal_conductivity')),
                               variant='backward')
        self.a_qy_t = self.d_y(factors=(1 / self.y.increment *
                                        self.material_vector('thermal_conductivity')),
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


class ThermalMaterial:
    """Class for specification of thermal material parameters."""

    def __init__(self, heat_capacity, density, thermal_conductivity):
        """Class constructor. Default values for optional parameters create lossless medium.

        Args:
            heat_capacity: Specific heat capacity.
            density: Density.
            thermal_conductivity: Thermal conductivity coefficient.
        """

        self.heat_capacity = heat_capacity
        self.density = density
        self.thermal_conductivity = thermal_conductivity
