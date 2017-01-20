import matplotlib.pyplot as pp
import multiprocessing as mp
import numpy as np
import pylab as pl
import siprefix as si
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

        self._plot_queue = mp.Queue()
        self._x_axis_prefix, self._x_axis_factor, _ = si.autoscale(max(self.field.x.vector))
        self._t_prefix, self._t_factor, _ = si.autoscale(max(self.field.t.vector))

        self.axes = None
        self.plot_title = ''
        self.x_label = '$x$'
        self.time_precision = 2


class Animator1D(Animator):
    """Animator for one dimensional field simulation in pyFDs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.y_label = self.observed_component

    def _sim_function(self, queue):
        """Simulation function to be started as a separate process."""

        for ii in range(int(self.field.t.samples / self.steps_per_frame)):
            self.field.simulate(self.steps_per_frame)
            queue.put((self.field.t.vector[self.field.step],
                       getattr(self.field, self.observed_component).values))

        # put None when simulation finishes
        queue.put(None)

    def plot_region(self, region):
        """Shows the given region in the field plot."""

        if type(region) == reg.PointRegion:
            self.axes.plot(np.ones(2) * region.point_coordinates / self._x_axis_factor,
                           np.array([-1, 1]) * self.scale, color='black')
        elif type(region) == reg.LineRegion:
            self.axes.plot(np.ones(2) * region.line_coordinates[0] / self._x_axis_factor,
                           np.array([-1, 1]) * self.scale, color='black')
            self.axes.plot(np.ones(2) * region.line_coordinates[1] / self._x_axis_factor,
                           np.array([-1, 1]) * self.scale, color='black')
        else:
            raise TypeError('Unknown type in region list: {}'.format(type(region)))

    def start_simulation(self):
        """Starts the simulation with visualization."""

        pp.figure()
        self.axes = pp.gca()
        main_plot, = self.axes.plot([])
        self.axes.set_xlim(0, max(self.field.x.vector) / self._x_axis_factor)
        self.axes.set_ylim(-self.scale, self.scale)
        self.axes.set_xlabel('{0} / {1}m'.format(self.x_label, self._x_axis_prefix))
        self.axes.set_ylabel(self.y_label)
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

        sim_process = mp.Process(target=self._sim_function, args=(self._plot_queue,))
        sim_process.start()

        # wait for simulation initialization
        while self._plot_queue.empty():
            pl.pause(0.1)

        finished = False
        while not finished:
            # wait for new simulation result
            while self._plot_queue.empty():
                pl.pause(0.01)

            message = self._plot_queue.get()
            # simulation function sends None when simulation is complete
            if message is None:
                finished = True
            else:
                time, data = message
                self.axes.title.set_text('{title} $t$ = {time:.{prec}f} {prefix}s'
                                         .format(title=self.plot_title, time=time/self._t_factor,
                                                 prec=self.time_precision, prefix=self._t_prefix))
                main_plot.set_data(self.field.x.vector / self._x_axis_factor, data)
                pl.pause(self.frame_delay)

        pp.show()
