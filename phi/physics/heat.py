from .field.effect import effect_applied
from .field.util import diffuse
from .physics import Physics, StateDependency
from .domain import DomainState
from phi import struct
import warnings


@struct.definition()
class HeatTemperature(DomainState):

    def __init__(self, domain, temperature=0, diffusivity=0.1, name='heatEquation', **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))

    @struct.variable(dependencies=DomainState.domain, default=0)
    def temperature(self, temperature):
        return self.centered_grid('temperature', temperature, self.rank)

    @struct.constant(default=0.1)
    def diffusivity(self, diffusivity):
        return diffusivity

    def __add__(self, other):
        return self.copied_with(temperature=self.temperature + other)


class HeatDiffusion(Physics):

    def __init__(self, default_diffusivity=0.1, diffusivity=None):
        Physics.__init__(self, [StateDependency('effects', 'temperature_effect', blocking=True)])
        if diffusivity is not None:
            warnings.warn("Argument 'diffusivity' is deprecated, use 'default_diffusivity' instead.",
                          DeprecationWarning)
            default_viscosity = diffusivity
        self.default_diffusivity = default_diffusivity

    def step(self, temperature, dt=1.0, effects=()):
        if isinstance(temperature, HeatTemperature):
            return temperature.copied_with(temperature=self.step_temperature(temperature.temperature,
                                                                             dt, effects), age=temperature.age + dt)
        else:
            return self.step_temperature(temperature, dt, effects)

    def step_temperature(self, temperature, dt=1.0, effects=()):
        # pylint: disable-msg = arguments-differ
        temperature = diffuse(temperature, dt * self.default_diffusivity)
        for effect in effects:
            temperature = effect_applied(effect, temperature, dt)
        return temperature.copied_with(age=temperature.age + dt)
