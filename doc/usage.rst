Usage
=====


Basics
------

Since pyfds does not have a graphical interface to create geometry, apply boundary, etc. yet, we
use small python script files to setup and run simulations. To use pyfds we must first import it:

.. code-block:: python

    import pyfds

We would suggest placing the setup and simulation code in

.. code-block:: python

    if __name__ == '__main__':

This is not always necessary, but some parts of pyfds (especially the visualization) requires it.
As of now, only acoustic simulation is implemented, so we use it as an example. If one is to
implement new types of field simulation, these should follow the same pattern (if applicable).
We start by creating the main material for the field to simulate. Here, we use a fluid
similar to water:

.. code-block:: python

    water = pyfds.AcousticMaterial(sound_velocity=1500,
                                   density=1000,
                                   shear_viscosity=1e-3)

Note that all properties are given in basic metric units without prefix. Now we create the field
object, which is the central object for all following steps. Along with the main material, the
size of the field to be simulated and the number and size of time steps is to be supplied. There
are already several different field classes implemented. This is how some of them are instantiated:

.. code-block:: python

    field = pyfds.Acoustic1D(t_delta=1e-6, t_samples=1201,
                             x_delta=1e-3, x_samples=301,
                             material=water)

or

.. code-block:: python

    field = pyfds.Acoustic3DAxi(t_delta=1e-6, t_samples=1201,
                                x_delta=1e-3, x_samples=301,
                                y_delta=1e-3, y_samples=1001,
                                material=water)

or

.. code-block:: python

    field = pyfds.IdealGas1D(t_delta=1e-7, t_samples=140001,
                             x_delta=1e-3, x_samples=24001,
                             material=argon)

At this point we could already start the simulation, but as there are no input signals to excite
waves in the field, nothing would happen.


Boundary Conditions
-------------------

In each field object, you will find several properties of the type
:class:`~pyfds.fields.FieldComponent`. These represent the different components of the field to
be simulated. For an acoustic field in a fluid, these are a pressure field and a velocity field.
As the velocity is a vector, it has again as many component as the field has dimensions. So if
you create an instance of :class:`~pyfds.acoustics.Acoustic2D`, you will find in it the field
components `pressure`, `velocity_x`, and `velocity_y` (Note that nonlinear acoustic field have
an additional component `density`).

Boundary conditions as well as signals to be saved are defined for these field components. To
add boundary conditions or output signals, you can use the methods
:func:`~pyfds.fields.FieldComponent.add_boundary` or
:func:`~pyfds.fields.FieldComponent.add_output`. Alternatively you can append an object of the
type :class:`~pyfds.regions.Boundary` or :class:`~pyfds.regions.Output` to the respective lists
in the :class:`~pyfds.fields.FieldComponent` (`boundaries` or `outputs`). The arguments of
:func:`~pyfds.fields.FieldComponent.add_boundary` and
:func:`~pyfds.fields.FieldComponent.add_output` are the same as the constructor arguments for
the respective classes. The first argument for both classes is an instance of classes derived from
:class:`~pyfds.regions.Region`. These specify at which points the boundary is applied or the
output is recorded. You can create these region by calling the methods
:func:`~pyfds.fields.Field.get_point_region`, :func:`~pyfds.fields.Field1D.get_line_region`, or
:func:`~pyfds.fields.Field2D.get_rect_region` depending on what kind of region you want the
boundary or output applied to (point, line, or rectangle). So, adding a boundary to the pressure
 in a two-dimensional field, that crosses the field diagonally would look like this:

.. code-block:: python

    field.pressure.add_boundary(field.get_line_region(
    (0, 0, max(field.x.vector), max(field.y.vector))))

Note that we used the properties `x` and `y` to avoid entering the coordinate directly. Note
also the the format of the coordinates to be entered in the `get_*_region` methods changes
depending on the dimensionality of the field. The class :class:`~pyfds.regions.Boundary`
respectively the method :func:`~pyfds.fields.FieldComponent.add_boundary` has a second argument
called value. By default, this is 0, meaning the values of the field component are kept 0 at the
specified region, resulting in Dirichlet type boundary condition. Alternatively you can supply a
signal in form of an numpy array that is the applied to the region sample after sample for each
simulation step. Note that the supplied array must at least be `field.t.samples` in length.

.. code-block:: python

    field.velocity_x.add_boundary(field.get_point_region(
    (0.01, 0.01)), value=some_signal)

You can also supply a list of numpy arrays if you what different signals applied to each point in
the region. The length of the list has to be the same as the number of points in the region.
There is a third argument in :func:`~pyfds.fields.FieldComponent.add_boundary` called
`additive`, which is False by default. Setting this to True results in the value of the
boundary, e.g. the signal, to be added to the field value at the specified region instead of
being set directly.


Output signals
--------------

Marking specific point to be saved as output signals basically works the same as creating a
fixed boundary:

.. code-block:: python

    field.pressure.add_output(field.get_line_region(
    (0, 0, max(field.x.vector), max(field.y.vector))))

You can then find the output signals (each point is saved separately) in `field.{component}
.output[{number of output region}].signals` as a list of arrays. There is also an additional
property `mean_signal` in the class :class:`~pyfds.regions.Output`, that returns the ensemble
average of all signals in the object.


Materials
---------

There is also the option to specify different materials using the same
:class:`~pyfds.regions.Region` classes as for boundaries and output signals. In each field
class, there is a method called :func:`~pyfds.fields.Field.add_material_region`. Alternatively
you can again append an object of type :class:`~pyfds.regions.MaterialRegion` to the property
`material_regions` of the field object. The first entry of this list is the main material
supplied when creating the field object. The constructor of
:class:`~pyfds.regions.MaterialRegion` as well as
:func:`~pyfds.fields.Field.add_material_region` take two arguments: The region the material is
specified for and the material itself:

.. code-block:: python

    field.add_material_region(field.get_rect_region((0.1, 0.1, 0.1, 0.1)),
                              material=pyfds.AcousticMaterial(1000, 500))


Starting the simulation
-----------------------

Before starting the simulation, you can check if the simulation will be stable:

.. code-block:: python

    print(field.is_stable())

While this does not guarantee that the simulation will not destabilize, it give an estimate and
can be used to rule out basic errors when choosing simulation parameters like quantization steps.
You can then start the simulation run using

.. code-block:: python

    field.simulate()

and hope for illuminating results.


Visualization
-------------

pyfds also provides some basic visualization tools, that can be used for on-the-fly field plots
during the simulation. This module still has some issues to be fixed. Due to performance
reasons the animator runs the simulation in a different process so a placement in
`if __name__ == '__main__':` or a similar construct is required. To visualize a simulation
process, just create an object of type :class:`~pyfds.gfx.Animator1D` or
:class:`~pyfds.gfx.Animator2D`, depending on the number of dimensions of the field you are going
to simulate. The animator's first constructor argument is the field object to simulate. The
second argument is the name of the field component to be observed as a string.

.. code-block:: python

    animator = fds.Animator2D(flied, observed_component='velocity_x')

There are several other optional argument of the animator's constructor as well as properties of
the animator that can be used to customize the resulting plot. To start the simulation with
visualization you have to start the simulation from the animator object by calling:

.. code-block:: python

    animator.start_simulation()

A matplotlib window should then open, displaying the field distribution. 