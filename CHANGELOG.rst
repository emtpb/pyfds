*********
Changelog
*********

This project follows the guidelines of `Keep a changelog`_ and adheres to
`Semantic versioning`_.

.. _Keep a changelog: http://keepachangelog.com/
.. _Semantic versioning: https://semver.org/


`Unreleased`_
=============


`0.2.0`_ - 2023-06-30
=====================

Added
-----
* Support for elliptic regions.

Fixed
-----
* Code style and documentation issues.
* Boundary test cases.

Changed
-------
* From setup.py to pyproject.toml.


`0.1.6`_ - 2023-03-30
=====================

Added
-----
* Acoustic simulation with moving medium.
* Asymmetric scaling for animations.

Fixed
-----
* Minor issues.


`0.1.5`_ - 2021-12-20
=====================

Added
-----
* Electrostatic field simulations.
* Thermal and electrostatics API documentation.

Fixed
-----
* Documentation issues.


`0.1.4`_ - 2021-11-02
=====================

Added
-----
* Thermal field simulations.
* Reset method that allows multiple simulation runs using the same Field object.

Fixed
-----
* Documentation issues.


`0.1.3`_ - 2020-02-21
=====================

Added
-----
* This changelog.
* A DOI tag.

Removed
-------
* Probably broken option to set multiple points in a boundary to different, constant values.

Fixed
-----
* Animator docstring.
* Stepping bug in segmented simulation runs.


`0.1.2`_ - 2019-04-24
=====================

Added
-----
* Saving animations as video files.
* Support for triangular regions.


`0.1.1`_ - 2018-07-16
=====================

Added
-----
* Preview function for the simulation setup.
* Logging.
* Setting acoustic losses directly.

Changed
-------
* Implementation of the simulation process: New simulation cases now only need to implement a method for a single time step.


`0.1.0`_ - 2017-06-21
=====================

Added
-----
* First preview release.


.. _Unreleased: https://github.com/emtpb/pyfds
.. _0.2.0: https://github.com/emtpb/pyfds/releases/tag/0.2.0
.. _0.1.6: https://github.com/emtpb/pyfds/releases/tag/0.1.6
.. _0.1.5: https://github.com/emtpb/pyfds/releases/tag/0.1.5
.. _0.1.4: https://github.com/emtpb/pyfds/releases/tag/0.1.4
.. _0.1.3: https://github.com/emtpb/pyfds/releases/tag/0.1.3
.. _0.1.2: https://github.com/emtpb/pyfds/releases/tag/0.1.2
.. _0.1.1: https://github.com/emtpb/pyfds/releases/tag/0.1.1
.. _0.1.0: https://github.com/emtpb/pyfds/releases/tag/0.1.0
