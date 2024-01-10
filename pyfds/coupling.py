import logging as lo

from . import fields as fld

logger = lo.getLogger('pyfds')


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


class MaterialCoupling():
    """Class to couple the material parameters of one field to a field quantity of another.

    Changing material properties during simulations require the system matrices to be reassembled
    during runtime. To avoid doing this in every time step use the parameter rel_change_threshold
    and stepping. Increasing stepping will trigger the reassembly only every nth time step.
    Increasing rel_change_threshold will trigger reassembly only when a given relative change from
    the previous value, e.g. 1e-3, is observed.
    """

    def __init__(self, source_component, target_field, target_parameter,
                 transfer_function, rel_change_threshold=None, stepping=1):
        """Class constructor.

        Args:
            source_component: Field component whose values are fed into the transfer function.
            target_field: Field with material to be modified.
            target_parameter: Name of parameter modified by the output of transfer function.
            transfer_function: Callable function that describes the interaction. Transfer function
                takes source field component values as parameter. Target material parameter is
                multiplied by the output of this function.
            rel_change_threshold: Relative change threshold for parameters changes to be applied.
            stepping: Increase to apply coupling only each nth step.
        """

        self.source_component = source_component
        self.target_field = target_field
        self.target_parameter = target_parameter
        self.transfer_function = transfer_function
        self.rel_change_threshold = rel_change_threshold
        self.stepping = stepping

        self.last_used_factors = 0

        """Replace the material_vector method of the target field with a method that accounts for
        the influence of the source field."""
        self.target_field.static_material_vector = self.target_field.material_vector

        def material_vector(mat_parameter):
            """Get a vector that contains the specified material parameter for every point of the
            field and multiply by the transfer function if the parameter is coupled to the source
            field.

            Args:
                mat_parameter: Material parameter of interest.

            Returns:
                Vector which contains the specified material parameter for each point in the field.
            """
            if mat_parameter == self.target_parameter:
                return self.target_field.static_material_vector(mat_parameter) \
                    * self.transfer_function(self.source_component.values)
            else:
                return self.target_field.static_material_vector(mat_parameter)

        self.target_field.material_vector = material_vector

    def apply(self, step):
        """Apply the interaction. Call at every simulation step.

        Args:
            step: Time step of the simulation.
        """

        if step % self.stepping == 0:
            transfer_factors = self.transfer_function(self.source_component.values)
            rel_change = max(abs((transfer_factors - self.last_used_factors) / transfer_factors))
            if self.rel_change_threshold is None or rel_change > self.rel_change_threshold:
                # Assemble_matrices calls the modified material_vector method.
                self.target_field.assemble_matrices()
                self.last_used_factors = transfer_factors
                if self.rel_change_threshold is not None:
                    logger.info(f"Relative change in parameters is {rel_change}.")
                    logger.info(f"Matrices reassembled in step {step}.")
