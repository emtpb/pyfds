from . import acoustics as ac
from . import coupling as cp
from . import thermal as th

__all__ = [
    'ThermoAcoustic1D',
]


class ThermoAcoustic1D(cp.SynchronizedFields):
    """Class for thermal and acoustic co-simulation with coupling due to acoustic losses based on
    the conservation of energy."""

    def __init__(self, x_samples, x_delta, t_samples, t_delta,
                 thermal_material, acoustic_material, stepping=1):
        """Class constructor.

        Args:
            x_samples: Number of samples in x direction.
            x_delta: Increment in x direction.
            t_samples: Number of time samples.
            t_delta: Time increment.
            thermal_material: Main material of the thermal field.
            acoustic_material: Main material of the acoustic field.
            stepping: Increase to apply coupling only each nth step.
        """

        acoustic_field = ac.Acoustic1D(x_samples, x_delta, t_samples, t_delta, acoustic_material)
        thermal_field = th.Thermal1D(x_samples, x_delta, t_samples, t_delta, thermal_material)

        acoustic_loss = cp.BoundaryCoupling(
            source_component=acoustic_field.velocity,
            target_component=thermal_field.temperature,
            transfer_function=self._loss_coupling,
            additive=True,
            accumulate=True if stepping > 1 else False,
            stepping=stepping
        )
        super().__init__([acoustic_field, thermal_field], [acoustic_loss])

        # add references to 1D specific methods for convenience
        self.get_index = acoustic_field.get_index
        self.get_position = acoustic_field.get_position
        self.get_line_region = acoustic_field.get_line_region

    def _loss_coupling(self, velocity):
        """Coupling of velocity and temperature field based on losses caused by viscosity.

        Args:
            velocity: Velocity field values.

        Returns:
            Temperature values to add to the temperature field.
        """

        """Spatial derivative of velocity is required. Reuse matrix from acoustic field but remove
        density factor present in original matrix."""
        velocity_derivative = self.fields[0].a_v_p.dot(
            velocity * self.fields[0].material_vector('density')
        )

        return self.fields[0].material_vector('absorption_coef') \
            / self.fields[1].material_vector('density') \
            / self.fields[1].material_vector('heat_capacity') \
            * velocity_derivative ** 2 * self.t.increment
