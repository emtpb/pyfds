Acoustic field simulation
=========================

One-dimensional acoustic simulation
-----------------------------------

.. code-block:: python

    import pyfds as fds
    import scipy.signal as sg
    import matplotlib.pyplot as pp

    # create material
    some_fluid = fds.AcousticMaterial(sound_velocity=700, density=0.01, shear_viscosity=1e-3)

    # create field
    fld = fds.Acoustic1D(t_delta=1e-7, t_samples=30001,
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

    # specify node to save output signals at (at x = 2 m)
    fld.pressure.outputs.append(fds.Output(fld.get_point_region(2)))

    # estimate if simulation is stable
    print(fld.is_stable())

    # start simulation
    fld.simulate()

    # plot output signal
    pp.figure()
    pp.plot(fld.pressure.outputs[0].signals[0])
    pp.show()


Three-dimensional, axial-symmetric acoustic simulation
------------------------------------------------------

Simulating a short, circular acoustic waveguide.

.. code-block:: python

    import pyfds as fds
    import matplotlib.pyplot as pp
    import scipy.signal as sg

    # placement in __main__ is only necessary if visualization is used
    if __name__ == '__main__':

        # create field
        fld = fds.Acoustic3DAxi(t_delta=1e-6, t_samples=1201,
                                x_delta=1e-3, x_samples=301,
                                y_delta=1e-3, y_samples=901,
                                material=fds.AcousticMaterial(sound_velocity=700,
                                                              density=1000,
                                                              shear_viscosity=1e-2))

        # generate excitation signal
        ex_signal = sg.gausspulse(fld.t.vector - 1.2e-4, int(50e3), 0.6)

        # add a linear boundary at right side to prevent the waves from warping around
        fld.velocity_x.add_boundary(fld.get_line_region((0, 0, 0, max(fld.y.vector))))

        # add waveguide boundary
        fld.velocity_x.boundaries.append(fds.Boundary(
            fld.get_line_region((200e-3, 0, 200e-3, max(fld.y.vector)))))

        # add excitation signal as boundary (do not excite at r=0 in axial symmetric setups)
        fld.pressure.boundaries.append(fds.Boundary(
            fld.get_line_region((1e-3, 200e-3, 199e-3, 200e-3)),
            value=ex_signal, additive=True))

        # add output positions
        fld.pressure.outputs.append(fds.Output(fld.get_line_region((0, 700e-3, 200e-3, 700e-3))))

        # uncomment these to visualize simulation process
        # anim = fds.Animator2D(field=fld, observed_component='pressure')
        # anim.start_simulation()

        # start simulation
        fld.simulate()

        # plot results
        pp.figure()
        pp.plot(fld.t.vector, ex_signal, label='Send signal')
        pp.plot(fld.t.vector, fld.pressure.outputs[0].mean_signal, label='Received signal')
        pp.xlabel('Time $t$ / s')
        pp.ylabel('Acoustic pressure / a.u.')
        pp.grid()
        pp.legend()

        pp.show()
