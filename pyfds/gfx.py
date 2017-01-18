import matplotlib.pyplot as pp
import multiprocessing as mp
import pylab as pl
from . import fields as fld


class Animator:
    """Base class for pyFDs' live field animation during simulation."""

    def __init__(self, field, observed_component=None, steps_per_frame=10, scale=1,
                 frame_delay=1e-2):

        self.field = field
        self.field_components = {name: getattr(self.field, name) for name in dir(self.field)
                                 if type(getattr(self.field, name)) == fld.FieldComponent}
        if observed_component:
            if observed_component in self.field_components.keys():
                self.observed_component = observed_component
            else:
                raise KeyError('Field component {} not found in given field.'
                               .format(observed_component))
        else:
            self.observed_component = list(self.field_components.keys())[0]

        self.steps_per_frame = int(steps_per_frame)
        self.scale = scale
        self.frame_delay = frame_delay

        self.show_boundaries = True
        self.show_materials = True
        self.show_output = True
        self.plot_queue = mp.Queue()


class Animator1D(Animator):
    """Animator for one dimensional field simulation in pyFDs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.plot = None

    @staticmethod
    def _sim_function(field, steps_per_frame, observed_component, queue):
        """Simulation function to be started as a separate process."""

        for ii in range(int(field.t.samples / steps_per_frame)):
            field.simulate(steps_per_frame)
            queue.put(getattr(field, observed_component).values)

        # put None when simulation finishes
        queue.put(None)

    def start_simulation(self):
        """Starts the simulation with visualization."""

        pp.figure()
        self.plot, = pp.plot([])
        self.plot.axes.set_xlim(0, max(self.field.x.vector))
        self.plot.axes.set_ylim(-self.scale, self.scale)

        sim_process = mp.Process(target=self._sim_function,
                                 args=(self.field, self.steps_per_frame,
                                       self.observed_component, self.plot_queue,))
        sim_process.start()

        # wait for simulation initialization
        while self.plot_queue.empty():
            pl.pause(0.1)

        finished = False
        while not finished:
            # wait for new simulation result
            while self.plot_queue.empty():
                pl.pause(0.01)

            data = self.plot_queue.get()
            # simulation function sends None when simulation is complete
            if data is None:
                finished = True
            else:
                self.plot.set_data(self.field.x.vector, data)
                pl.pause(self.frame_delay)

        pp.show()
