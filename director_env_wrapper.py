import numpy as np

class DirectorEnvWrapper(object):
    metadata = {'render.modes': []}
    reward_range = (-np.inf, np.inf)
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)