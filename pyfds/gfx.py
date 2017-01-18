import matplotlib.pyplot as pp
import multiprocessing as mp
import numpy as np
import pylab as pl
from . import fields as fld
from . import regions as reg


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

    def _sim_function(self, queue):
        """Simulation function to be started as a separate process."""

        for ii in range(int(self.field.t.samples / self.steps_per_frame)):
            self.field.simulate(self.steps_per_frame)
            queue.put(getattr(self.field, self.observed_component).values)

        # put None when simulation finishes
        queue.put(None)

    def plot_region(self, region):
        """Shows the given region in the field plot."""

        if type(region) == reg.PointRegion:
            pp.plot(np.ones(2) * region.point_coordinates,
                    np.array([-1, 1]) * self.scale, color='black')
        elif type(region) == reg.LineRegion:
            pp.plot(np.ones(2) * region.line_coordinates[0],
                    np.array([-1, 1]) * self.scale, color='black')
            pp.plot(np.ones(2) * region.line_coordinates[1],
                    np.array([-1, 1]) * self.scale, color='black')
        else:
            raise TypeError('Unknown type in region list: {}'.format(type(region)))

    def start_simulation(self):
        """Starts the simulation with visualization."""

        pp.figure()
        self.plot, = pp.plot([])
        self.plot.axes.set_xlim(0, max(self.field.x.vector))
        self.plot.axes.set_ylim(-self.scale, self.scale)
        pp.grid(True)

        if self.show_materials:
            for mat_region in self.field.material_regions:
                self.plot_region(mat_region.region)

        if self.show_boundaries:
            for name, component in self.field_components.items():
                for boundary in component.boundaries:
                    self.plot_region(boundary.region)

        if self.show_output:
            for name, component in self.field_components.items():
                for output in component.outputs:
                    self.plot_region(output.region)

        sim_process = mp.Process(target=self._sim_function, args=(self.plot_queue,))
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
