import sys

import pytest

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

def setup_module(_):
    env = SeedEnv()
    env_id = env.__class__.__name__ + '-v0'
    entry = env.__class__.__module__ + ':' + env.__class__.__name__
    register(env_id, entry_point=entry)


def test_multiprocessing_env_seed_propagates():
    venv = MultiprocessingEnv('SeedEnv-v0')
    venv.configure(n=4)
    venv.seed([1, 2, 3, 4])
    assert venv.reset() == [1, 2, 3, 4]

if __name__ == '__main__':
    pytest.main(sys.argv)
