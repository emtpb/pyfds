from . import fields as fld


class Acoustic1D(fld.Field1D):
    """Class for simulation of one dimensional acoustic fields."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pressure = fld.FieldComponent(self.num_points)
        self.velocity = fld.FieldComponent(self.num_points)

        # initialize attributes sparse matrices
        self.a_pv = None
        self.a_vp = None
        self.a_vv = None

    def simulate(self):
        """Starts the simulation."""

        self.a_pv = self.d_x(factors=(self.t.increment / self.x.increment *
                                      self.material_vector('sound_velocity') ** 2 *
                                      self.material_vector('density')))
        self.a_vp = self.d_x(factors=(self.t.increment / self.x.increment /
                                      self.material_vector('density')), backward=True)
        self.a_vv = self.d_x2(factors=(self.t.increment / self.x.increment ** 2 *
                                       self.material_vector('absorption_coef') /
                                       self.material_vector('density')))

        for ii in range(self.t.samples):

            self.pressure.apply_bounds(ii)
            self.pressure.write_outputs()

            self.velocity.values -= (self.a_vp.dot(self.pressure.values) -
                                     self.a_vv.dot(self.velocity.values))

            self.velocity.apply_bounds(ii)
            self.velocity.write_outputs()

            self.pressure.values -= self.a_pv.dot(self.velocity.values)


class Acoustic2D(fld.Field2D):
    """Class for simulation of two dimensional acoustic fields."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pressure = fld.FieldComponent(self.num_points)
        self.velocity_x = fld.FieldComponent(self.num_points)
        self.velocity_y = fld.FieldComponent(self.num_points)

        # initialize attributes sparse matrices
        self.a_p_vx = None
        self.a_p_vy = None
        self.a_vx_p = None
        self.a_vy_p = None
        self.a_vx_vx = None
        self.a_vy_vy = None

    def simulate(self):
        """Starts the simulation."""

        self.a_p_vx = self.d_x(factors=(self.t.increment / self.x.increment *
                                        self.material_vector('sound_velocity') ** 2 *
                                        self.material_vector('density')))
        self.a_p_vy = self.d_y(factors=(self.t.increment / self.y.increment *
                                        self.material_vector('sound_velocity') ** 2 *
                                        self.material_vector('density')))
        self.a_vx_p = self.d_x(factors=(self.t.increment / self.x.increment /
                                        self.material_vector('density')), backward=True)
        self.a_vy_p = self.d_y(factors=(self.t.increment / self.y.increment /
                                        self.material_vector('density')), backward=True)
        self.a_vx_vx = (self.d_x2(factors=(self.t.increment / self.x.increment ** 2 *
                                           self.material_vector('absorption_coef') /
                                           self.material_vector('density'))) +
                        self.d_y2(factors=(self.t.increment / self.y.increment ** 2 *
                                           self.material_vector('absorption_coef') /
                                           self.material_vector('density'))))
        self.a_vy_vy = self.a_vx_vx

        for ii in range(self.t.samples):

            self.pressure.apply_bounds(ii)
            self.pressure.write_outputs()

            self.velocity_x.values -= (self.a_vx_p.dot(self.pressure.values) -
                                       self.a_vx_vx.dot(self.velocity_x.values))
            self.velocity_y.values -= (self.a_vy_p.dot(self.pressure.values) -
                                       self.a_vy_vy.dot(self.velocity_y.values))

            self.velocity_x.apply_bounds(ii)
            self.velocity_x.write_outputs()
            self.velocity_y.apply_bounds(ii)
            self.velocity_y.write_outputs()

            self.pressure.values -= (self.a_p_vx.dot(self.velocity_x.values) +
                                     self.a_p_vy.dot(self.velocity_y.values))


class AcousticMaterial:
    """Class for specification of acoustic material parameters."""

    def __init__(self, sound_velocity, density,
                 shear_viscosity=0, bulk_viscosity=0,
                 thermal_conductivity=0, isobaric_heat_cap=1, isochoric_heat_cap=1):
        """Default values for optional parameters create lossless medium."""

        self.sound_velocity = sound_velocity
        self.density = density
        self.shear_viscosity = shear_viscosity
        self.bulk_viscosity = bulk_viscosity
        self.thermal_conductivity = thermal_conductivity
        self.isobaric_heat_cap = isobaric_heat_cap
        self.isochoric_heat_cap = isochoric_heat_cap

    @property
    def absorption_coef(self):
        """This is a helper variable that sums up all losses into a single quantity."""

        return (4/3 * self.shear_viscosity + self.bulk_viscosity + self.thermal_conductivity *
                (self.isobaric_heat_cap - self.isochoric_heat_cap) /
                (self.isobaric_heat_cap * self.isochoric_heat_cap))
