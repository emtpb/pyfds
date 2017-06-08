Acoustic field simulation
=========================

One dimensional acoustic simulation
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
