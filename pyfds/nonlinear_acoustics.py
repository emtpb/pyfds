import numpy as np
from . import fields as fld
from . import acoustics as ac


class IdealGas1D(fld.Field1D):
    """Class for simulation of one dimensional nonlinear acoustic fields in ideal gases."""

    def __init__(self, convective=True, nl_state=True, *args, **kwargs):
        """Class constructor.

        Args:
            convective: Account for convective terms.
            nl_adiabatic: Account for nonlinear equation of state.
            See pyfds.fields.Field1D constructor arguments.
        """

        super().__init__(*args, **kwargs)
        self.convective = convective
        self.nl_state = nl_state
        self.pressure = fld.FieldComponent(self.num_points)
        self.velocity = fld.FieldComponent(self.num_points)
        self.density = fld.FieldComponent(self.num_points)

        # initialize attributes sparse matrices
        self.a_d_v = None
        self.a_v_p = None
        self.a_v_v = None
        self.a_v_v2 = None

    def create_matrices(self):
        """Creates the a_* matrices required for simulation."""

        self.a_d_v = self.d_x(factors=(self.t.increment / self.x.increment *
                                       np.ones(self.x.samples)))
        self.a_v_p = self.d_x(factors=(self.t.increment / self.x.increment) *
                              np.ones(self.x.samples), variant='backward')
        self.a_v_v = self.d_x2(factors=(self.t.increment / self.x.increment ** 2 *
                                        self.material_vector('absorption_coef')))
        self.a_v_v2 = self.d_x(factors=(self.t.increment / self.x.increment / 2) *
                               np.ones(self.x.samples), variant='central')

    def simulate(self, num_steps=None):
        """Starts the simulation.

        Args:
            num_steps: Number of steps to simulate (self.t.samples by default).
        """

        if not num_steps:
            num_steps = self.t.samples

        # create a_* matrices if create_matrices was not called before
        if self.a_d_v is None or self.a_v_p is None or self.a_v_v is None or self.a_v_v2 is None:
            self.create_matrices()

        # buffer static density and pressure, heat capacity ratio for easier access
        density = self.material_vector('density')
        sound_velocity = self.material_vector('sound_velocity')
        heat_cap_ratio = (self.material_vector('isobaric_heat_cap') /
                          self.material_vector('isochoric_heat_cap'))

        start_step = self.step
        for self.step in range(start_step, start_step + num_steps):

            self.pressure.apply_bounds(self.step)
            self.pressure.write_outputs()

            if self.convective:
                self.velocity.values -= (self.a_v_p.dot(self.pressure.values) /
                                         (density + self.density.values) +
                                         self.a_v_v2.dot(self.velocity.values) -
                                         self.a_v_v.dot(self.velocity.values) /
                                         (density + self.density.values))
            else:
                self.velocity.values -= (self.a_v_p.dot(self.pressure.values) /
                                         (density + self.density.values) -
                                         self.a_v_v.dot(self.velocity.values) /
                                         (density + self.density.values))

            self.velocity.apply_bounds(self.step)
            self.velocity.write_outputs()

            self.density.values -= self.a_d_v.dot((self.density.values + density) *
                                                  self.velocity.values)

            self.density.apply_bounds(self.step)
            self.density.write_outputs()

            if self.nl_state:
                # using modified equation of state, so static pressure is not required
                self.pressure.values = density * sound_velocity**2 / heat_cap_ratio * \
                    (((density + self.density.values) / density)**heat_cap_ratio - 1)
            else:
                self.pressure.values = sound_velocity**2 * self.density.values

    def is_stable(self):
        """Checks if simulation satisfies stability conditions. Does not account for instability
        due to high absorption or nonlinear effects. Includes a little headroom (1%).

        Returns:
            True if stable, False if not.
        """

        return np.all(self.material_vector('sound_velocity') <
                      0.99 * self.x.increment / self.t.increment)


class Acoustic2ndOrder1D(IdealGas1D):
    """Class for simulation of one dimensional nonlinear acoustic fields using second order 
    approximation.
    
    Args:
        See pyfds.fields.Field1D constructor arguments.
    """

    def simulate(self, num_steps=None):
        """Starts the simulation.

        Args:
            num_steps: Number of steps to simulate (self.t.samples by default).
        """

        if not num_steps:
            num_steps = self.t.samples

        # create a_* matrices if create_matrices was not called before
        if self.a_d_v is None or self.a_v_p is None or self.a_v_v is None or self.a_v_v2 is None:
            self.create_matrices()

        # buffer material vectors for better performance
        density = self.material_vector('density')
        d_rho_p = self.material_vector('d_rho_p')
        d_rho2_p = self.material_vector('d_rho2_p')

        start_step = self.step
        for self.step in range(start_step, start_step + num_steps):

            self.pressure.apply_bounds(self.step)
            self.pressure.write_outputs()

            self.velocity.values -= (self.a_v_p.dot(self.pressure.values) /
                                     (density + self.density.values) +
                                     self.a_v_v2.dot(self.velocity.values) -
                                     self.a_v_v.dot(self.velocity.values) /
                                     (density + self.density.values))

            self.velocity.apply_bounds(self.step)
            self.velocity.write_outputs()

            self.density.values -= self.a_d_v.dot((self.density.values + density) *
                                                  self.velocity.values)

            self.density.apply_bounds(self.step)
            self.density.write_outputs()

            self.pressure.values = d_rho_p * self.density.values + \
                d_rho2_p / 2 * self.density.values**2


class AcousticMaterial2ndOrder(ac.AcousticMaterial):
    """Class for specification of acoustic material parameters."""

    def __init__(self, d_rho_p=None, d_rho2_p=0, *args, **kwargs):
        """Class constructor. Default values for optional parameters create lossless medium.

        Args:
            d_rho_p: First derivative of the pressure with respect to density.
            d_rho2_p: Second derivative of the pressure with respect to density.
        """

        super().__init__(*args, **kwargs)
        if not d_rho_p:
            self.d_rho_p = super().sound_velocity**2
        else:
            self.d_rho_p = d_rho_p
        self.d_rho2_p = d_rho2_p
