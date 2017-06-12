Visualization
=============

Visualizing one dimensional simulations
---------------------------------------

.. code-block:: python

    import pyfds as fds
    import scipy.signal as sg

    # placement in __main__ is required as animation used multiprocessing
    if __name__ == '__main__':

        # create material
        some_fluid = fds.AcousticMaterial(sound_velocity=700, density=0.01, shear_viscosity=1e-3)

        # create field
        fld = fds.Acoustic1D(t_delta=1e-7, t_samples=100001,
                             x_delta=1e-3, x_samples=3001,
                             material=some_fluid)

        # generate some signal for excitation
        ex_signal = sg.gausspulse(fld.t.vector - 0.3e-3, int(10e3), 0.7)

        # create boundary for velocity and pressure at the beginning and end of field
        # (respective field quantity is set to 0)
        fld.velocity.boundaries.append(fds.Boundary(fld.get_point_region(0)))
        fld.pressure.boundaries.append(fds.Boundary(fld.get_point_region(max(fld.x.vector))))

        # apply the excitation signal at 1 m (has to be at least a long a fld.t.vector)
        fld.pressure.boundaries.append(fds.Boundary(fld.get_point_region(1),
                                                    value=ex_signal, additive=True))

        # create animator, supply field component to observe and number of steps in between
        # animation frames
        anim = fds.Animator1D(fld, observed_component='velocity', steps_per_frame=100)

        # start simulation with animation
        anim.start_simulation()


Visualizing two dimensional simulations
---------------------------------------

.. code-block:: python

    import pyfds as fds
    import scipy.signal as sg

    # placement in __main__ is required as animation used multiprocessing
    if __name__ == '__main__':

        # create field
        fld = fds.Acoustic2D(t_delta=1e-6, t_samples=2000,
                             x_delta=1e-3, x_samples=401,
                             y_delta=1e-3, y_samples=301,
                             material=fds.AcousticMaterial(sound_velocity=700,
                                                           density=1000,
                                                           shear_viscosity=1e-2))

        # generate some signal for excitation
        ex_signal = sg.gausspulse(fld.t.vector - 1.2e-4, int(30e3), 0.6)

        # add angled boundary line for excitation
        fld.pressure.add_boundary(fld.get_line_region((0.05, 0.01, 0.1, 0.05)),
                                  value=ex_signal, additive=True)

        # add a rectangular region with a different material
        fld.add_material_region(fld.get_rect_region((0.1, 0.1, 0.1, 0.1)),
                                material=fds.AcousticMaterial(200, 1000))

        # add a linear boundary at right side to prevent the waves from warping around
        fld.velocity_x.add_boundary(fld.get_line_region((0, 0, 0, max(fld.y.vector))))

        # create animator (observe pressure, update every 10th simulation set, color map limit 0.5)
        anim = fds.Animator2D(field=fld, observed_component='pressure', steps_per_frame=10, scale=0.5)

        # start simulation with animation
        anim.start_simulation()

