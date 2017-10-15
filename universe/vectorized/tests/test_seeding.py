import unittest

from gym import Env
from gym.envs import register
from universe.vectorized import MultiprocessingEnv


class SeedEnv(Env):
    def __init__(self):
        super().__init__()
        self._seed_value = None

    def _seed(self, seed=None):
        self._seed_value = seed
        return [seed]

    def _reset(self):
        return self._seed_value


class TestMultiProcessingSeedEnv(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_id = None

    def setUp(self):
        super().setUp()
        env = SeedEnv()
        self.env_id = env.__class__.__name__ + '-v0'
        entry = env.__class__.__module__ + ':' + env.__class__.__name__
        register(self.env_id, entry_point=entry)

    def test_multiprocessing_env_seed_propagates(self):
        venv = MultiprocessingEnv(self.env_id)
        venv.configure(n=4)
        venv.seed([1, 2, 3, 4])
        self.assertEqual(venv.reset(), [1, 2, 3, 4])

if __name__ == '__main__':
    unittest.main()
