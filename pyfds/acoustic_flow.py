import logging as lo
import numpy as np
import warnings as wn
from . import acoustics as acs

__all__ = [
    'AcousticFlow2D',
]

logger = lo.getLogger('pyfds')


class AcousticFlow2D(acs.Acoustic2D):
    """Class for simulation of two-dimensional acoustic fields with moving medium.

    The implementation is an approximation realised by moving the values in the field components
    along the x axis as specified by the flow velocity.
    """

    def __init__(self, flow, *args, **kwargs):
        """Class constructor.

        Args:
            flow: Flow velocity in x direction (scalar or vector with length of y_samples).
            *args, \\**kwargs: See pyfds.fields.Field2D constructor arguments.
        """
        super().__init__(*args, **kwargs)

        if isinstance(flow, (list, np.ndarray)) and \
                len(flow) == self.y.samples:
            self.flow = np.asarray(flow)
        elif isinstance(flow, (float, int)):
            self.flow = np.ones(self.y.samples) * flow
        else:
            raise ValueError('Flow must either be scalar or a vector with length of y_samples.')

        # Determine at which time step period the values have to be shifted.
        self.flow_t_deltas = (self.x.increment / self.flow / self.t.increment).astype(int)

        if np.any(self.flow_t_deltas == 1) or np.any(self.flow_t_deltas == 0):
            wn.warn('Flow velocity may be to high. Consider reducing t_delta.', stacklevel=2)
            logger.warning('Flow velocity may be to high. Consider reducing t_delta.')

    def sim_step(self):
        """Simulate one step."""
        super().sim_step()
        self.apply_flow()

    def apply_flow(self):
        """Apply flow field to the field components."""

        for component in [self.pressure, self.velocity_x, self.velocity_y]:
            for n, f in enumerate(self.flow_t_deltas):
                if self.step % f == 0:
                    component.values[n * self.x.samples + 1: (n + 1) * self.x.samples] = \
                        component.values[n * self.x.samples: (n + 1) * self.x.samples - 1]
                    component.values[n * self.x.samples: n * self.x.samples + 1] = 0
