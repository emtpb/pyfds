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
