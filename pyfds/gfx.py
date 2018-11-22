import matplotlib.patches as pa
import matplotlib.pyplot as pp
import multiprocessing as mp
import numpy as np
import os
import pylab as pl
from . import fields as fld
from . import regions as reg


class Animator:
    """Base class for pyFDs' live field animation during simulation."""

    def __init__(self, field, observed_component=None, steps_per_frame=10, scale=1,
                 frame_delay=1e-2, save_video=False, video_file_name='pyfds_vid.mp4'):
        """Class constructor.

        Args:
            field: Field to be observed.
            observed_component: Component to be observed (as string).
            steps_per_frame: Simulation steps between updates of the animation.
            scale: Scale of the animation.
            frame_delay: Delay between animation updates.
            save_video: Save animation as video (requires ffmpeg).
            video_name: File name for the video.
        """

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
        self.save_video = save_video
        self.video_file_name = video_file_name
        self.image_files = []
        self.ffmpeg_path = 'ffmpeg'
        self.video_fps = 30
        self.video_dpi = 100

        self.show_boundaries = True
        self.show_materials = True
        self.show_output = True

        self._plot_queue = mp.Queue()
        self._x_axis_prefix, self._x_axis_factor = get_prefix(max(self.field.x.vector))
        self._t_prefix, self._t_factor = get_prefix(max(self.field.t.vector))

        self.axes = None
        self.plot_title = ''
        self.x_label = '$x$'
        self.time_precision = 2

    def _sim_function(self, queue):
        """Simulation function to be started as a separate process.

        Args:
            queue: Instance of multiprocessing.Queue that is used to transfer data between
                simulation and visualization process.
        """

        for ii in range(int(self.field.t.samples / self.steps_per_frame)):
            self.field.simulate(self.steps_per_frame)
            queue.put((self.field.t.vector[self.field.step],
                       getattr(self.field, self.observed_component).values))

        # return field when simulation finishes to get output signals
        queue.put(self.field)

    def _update_components(self, message):
        """Function to be called when simulation process finished to update the field components 
        the main process including the output signals
        
        Args:
            message: Field object returned by the simulation process.
        """

        for name in dir(self.field):
            if type(getattr(self.field, name)) == fld.FieldComponent:
                setattr(self.field, name, getattr(message, name))

    def _save_frame(self):
        """Save current frame of animation to a png file for video creation."""

        file_name = 'frame{0:06d}.png'.format(len(self.image_files))
        pl.savefig(file_name, dpi=self.video_dpi)
        self.image_files.append(file_name)

    def _create_video(self):
        """Create video file from saved png files."""

        answer = None
        if os.path.isfile(self.video_file_name):
            answer = input('File {} already exists. Overwrite? [y/N]'.format(self.video_file_name))
            if answer.capitalize() == 'Y':
                os.remove(self.video_file_name)

        if not answer or answer.capitalize() == 'Y':
            # video codec requires the image size to be dividable by 2. Numbers (6.4 and 4.78)
            # result from matplotlibs standard figure size.
            width = int((self.video_dpi * 6.4)//2 * 2)
            height = int((self.video_dpi * 4.78)//2 * 2)
            os.system('{} -loglevel error -framerate {} -i frame%06d.png -pix_fmt yuv420p -vf '
                      'scale={}:{} {}'.format(self.ffmpeg_path, self.video_fps, width, height,
                                              self.video_file_name))

        for file in self.image_files:
            os.remove(file)


class Animator1D(Animator):
    """Animator for one-dimensional field simulation in pyFDs."""

    def __init__(self, *args, **kwargs):
        """Class constructor.

        Args:
            See pyfds.gfx.Animator constructor arguments.
        """
        super().__init__(*args, **kwargs)

        self.y_label = self.observed_component

    def plot_region(self, region):
        """Shows the given region in the field plot.

        Args:
            region: Region to be plotted.
        """

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

    def show_setup(self, halt=True):
        """Open a plot window that shows the simulation setup including boundaries, outputs and 
        material regions.
        
        Args:
            halt: Halt script execution until plot window is closed.
        """

        pp.figure()
        self.axes = pp.gca()
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

        if halt:
            pp.show()

    def start_simulation(self):
        """Starts the simulation with visualization."""

        self.show_setup(halt=False)
        main_plot, = self.axes.plot([])

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
            # simulation function returns field object when simulation is complete to get output
            if isinstance(message, fld.Field):
                # update main process field components with simulation result
                self._update_components(message)
                finished = True
            else:
                time, data = message
                self.axes.title.set_text('{title} $t$ = {time:.{prec}f} {prefix}s'
                                         .format(title=self.plot_title, time=time/self._t_factor,
                                                 prec=self.time_precision, prefix=self._t_prefix))
                main_plot.set_data(self.field.x.vector / self._x_axis_factor, data)
                pl.pause(self.frame_delay)
                self._save_frame()

        sim_process.join()
        if self.save_video:
            self._create_video()
        pp.show()


class Animator2D(Animator):
    """Animator for two-dimensional field simulation in pyFDs."""

    def __init__(self, *args, **kwargs):
        """Class constructor.

        Args:
            See pyfds.gfx.Animator constructor arguments.
        """
        super().__init__(*args, **kwargs)

        self._y_axis_prefix = self._x_axis_prefix
        self._y_axis_factor = self._x_axis_factor

        self.y_label = '$y$'
        self.c_label = self.observed_component

    def plot_region(self, region):
        """Shows the given region in the field plot.

        Args:
            region: Region to be plotted.
        """

        if type(region) == reg.PointRegion:
            self.axes.scatter(region.point_coordinates[0] / self._x_axis_factor,
                              region.point_coordinates[1] / self._y_axis_factor, color='black')
        elif type(region) == reg.LineRegion:
            self.axes.plot([region.line_coordinates[0] / self._x_axis_factor,
                            region.line_coordinates[2] / self._x_axis_factor],
                           [region.line_coordinates[1] / self._y_axis_factor,
                            region.line_coordinates[3] / self._y_axis_factor],
                           color='black')
        elif type(region) == reg.RectRegion:
            self.axes.add_patch(pa.Rectangle((region.rect_coordinates[0] / self._x_axis_factor,
                                              region.rect_coordinates[1] / self._y_axis_factor),
                                             region.rect_coordinates[2] / self._x_axis_factor,
                                             region.rect_coordinates[3] / self._y_axis_factor,
                                             fill=False))
        elif type(region) == reg.TriRegion:
            self.axes.add_patch(pa.Polygon(
                np.array([[region.tri_coordinates[0] / self._x_axis_factor,
                           region.tri_coordinates[1] / self._y_axis_factor],
                          [region.tri_coordinates[2] / self._x_axis_factor,
                           region.tri_coordinates[3] / self._y_axis_factor],
                          [region.tri_coordinates[4] / self._x_axis_factor,
                           region.tri_coordinates[5] / self._y_axis_factor]]),
                fill=False))
        else:
            raise TypeError('Unknown type in region list: {}'.format(type(region)))

    def field_as_matrix(self, component=None):
        """Returns a field component (observed component by default) as a numpy matrix.

        Args:
            component: Field component to be returned as matrix.

        Returns:
            Field as matrix.
        """

        if component is None:
            component = getattr(self.field, self.observed_component).values

        return np.reshape(component, (self.field.y.samples, self.field.x.samples))

    def show_setup(self, halt=True):
        """Open a plot window that shows the simulation setup including boundaries, outputs and 
        material regions.

        Args:
            halt: Halt script execution until plot window is closed.
        """

        pp.figure()
        self.axes = pp.gca()
        pp.axis('equal')
        self.axes.set_xlim(0, max(self.field.x.vector) / self._x_axis_factor)
        self.axes.set_ylim(0, max(self.field.y.vector) / self._y_axis_factor)
        self.axes.set_xlabel('{0} / {1}m'.format(self.x_label, self._x_axis_prefix))
        self.axes.set_ylabel('{0} / {1}m'.format(self.y_label, self._y_axis_prefix))

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

        if halt:
            pp.show()

    def start_simulation(self):
        """Starts the simulation with visualization."""

        self.show_setup(halt=False)
        main_plot = self.axes.imshow(self.field_as_matrix(),
                                     extent=(0, max(self.field.x.vector) / self._x_axis_factor,
                                             max(self.field.y.vector) / self._y_axis_factor, 0),
                                     cmap='viridis')
        main_plot.set_clim(-self.scale, self.scale)
        color_bar = pp.colorbar(main_plot)
        color_bar.set_label(self.observed_component, rotation=270)

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
            # simulation function returns field object when simulation is complete to get output
            if isinstance(message, fld.Field):
                # update main process field components with simulation result
                self._update_components(message)
                finished = True
            else:
                time, data = message
                self.axes.title.set_text('{title} $t$ = {time:.{prec}f} {prefix}s'
                                         .format(title=self.plot_title, time=time/self._t_factor,
                                                 prec=self.time_precision, prefix=self._t_prefix))
                main_plot.set_data(self.field_as_matrix(data))
                pl.pause(self.frame_delay)
                self._save_frame()

        sim_process.join()
        if self.save_video:
            self._create_video()
        pp.show()


def get_prefix(value):
    """Determine the metric prefix for a given value.
    
    Args:
        value: Value to determine the prefix for.

    Returns:
        prefix: String specifying the prefix.
        factor: Scale factor of the prefix.        
    """
    for factor, prefix in sorted(prefixes.items()):
        if value / factor < 1e3:
            break
    return prefix, factor


prefixes = {
    1e-24: 'y',
    1e-21: 'z',
    1e-18: 'a',
    1e-15: 'f',
    1e-12: 'p',
    1e-9: 'n',
    1e-6: 'µ',
    1e-3: 'm',
    1e0: '',
    1e3: 'k',
    1e6: 'M',
    1e9: 'G',
    1e12: 'T',
    1e15: 'P',
    1e18: 'E',
    1e21: 'Z',
    1e24: 'Y',
}
