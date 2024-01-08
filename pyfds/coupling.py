from . import fields as fld


class SynchronizedFields(fld.Field):
    """Class for simulation of coupled fields with same time stepping."""

    def __init__(self, fields, interactions):
        """Class constructor.

        Args:
            fields: List of Field objects with same spatial and temporal resolution.
            interactions: List of Interaction objects.
        """

        self.fields = fields
        self.interactions = interactions

        """To enable visualisation with the gfx module, references to the necessary attributes
        are created:"""
        for field in self.fields:
            for name, value in vars(field).items():
                if isinstance(value, fld.FieldComponent):
                    if not hasattr(self, name):
                        self.__setattr__(name, value)
                    else:
                        raise RuntimeError("Coupling of fields with identically named components "
                                           "is currently not possible")

        self.t = self.fields[0].t
        self.x = self.fields[0].x
        if hasattr(self.fields[0], 'y'):
            self.y = self.fields[0].y

    @property
    def step(self):
        """Retrieve current step from first field in list."""
        return self.fields[0].step

    @step.setter
    def step(self, value):
        """Make sure all field are always on the same time step."""
        for field in self.fields:
            field.step = value

    @property
    def num_points(self):
        """Returns number of points in the field."""
        return self.fields[0].num_points

    @property
    def material_regions(self):
        """Make material region list read only. To add new regions, modify individual fields."""

        regions = []
        for field in self.fields:
            regions += field.material_regions
        return regions

    def assemble_matrices(self):
        """Assemble the matrices and vectors required for simulation."""

        for field in self.fields:
            field.assemble_matrices()

    @property
    def matrices_assembled(self):
        """Check if for simulation are assembled for all fields."""

        return all([field.matrices_assembled for field in self.fields])

    def sim_step(self):
        """Simulate one step."""

        for field in self.fields:
            field.sim_step()
        for interaction in self.interactions:
            interaction.apply(self.step)


class BoundaryCoupling():
    """Class to couple the excitation or boundaries of one field to the field quantity of
    another."""

    def __init__(self, source_component, target_component, transfer_function,
                 additive=True, accumulate=False, stepping=1):
        """Class constructor.

        Args:
            source_component: Field component whose values are fed into the transfer function.
            target_component: Field component that is modified by the output of transfer function.
            transfer_function: Callable function that describes the interaction.
            additive: Add output of transfer function to target component or set it directly.
            accumulate: If stepping > 1, accumulate output of the transfer function when target
                component is not modified.
            stepping: Increase to apply coupling only each nth step.
        """

        self.source_component = source_component
        self.target_component = target_component
        self.transfer_function = transfer_function
        self.additive = additive
        self.accumulate = accumulate
        self.stepping = stepping

        self.accumulated_transfer = 0

    def apply(self, step):
        """Apply the interaction. Call at every simulation step.

        Args:
            step: Time step of the simulation.
        """

        if self.accumulate is True:
            self.accumulated_transfer += self.transfer_function(self.source_component.values)

        if step % self.stepping == 0:
            if self.accumulate is False:
                if self.additive is True:
                    self.target_component.values += \
                        self.transfer_function(self.source_component.values)
                else:
                    self.target_component.values = \
                        self.transfer_function(self.source_component.values)
            else:
                if self.additive is True:
                    self.target_component.values += self.accumulated_transfer
                else:
                    self.target_component.values = self.accumulated_transfer
                self.accumulated_transfer = 0
