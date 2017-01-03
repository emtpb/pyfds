class AcousticMaterial:
    """Class for specification of acoustic material parameters."""

    def __init__(self, sound_velocity, density,
                 shear_viscosity=0, bulk_viscosity=0,
                 thermal_conductivity=0, isobaric_heat_cap=1, isochoric_heat_cap=1):
        """Default values for optional parameters create lossless medium."""

        self.sound_velocity = sound_velocity
        self.density = density
        self.shear_viscosity = shear_viscosity
        self.bulk_viscosity = bulk_viscosity
        self.thermal_conductivity = thermal_conductivity
        self.isobaric_heat_cap = isobaric_heat_cap
        self.isochoric_heat_cap = isochoric_heat_cap

    @property
    def absorption_coef(self):
        """This is a helper variable that sums up all losses into a single quantity."""

        return (4/3 * self.shear_viscosity + self.bulk_viscosity + self.thermal_conductivity *
                (self.isobaric_heat_cap - self.isochoric_heat_cap) /
                (self.isobaric_heat_cap * self.isochoric_heat_cap))
