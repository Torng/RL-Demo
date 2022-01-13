from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
class Myenv(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0,maximum=5, name='observation')
        self._state = 0
        self._episode_ended = False
        self.end_state = 6
        self.game_map = [0,0,0,0,0,'*']
        self.step_count = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec


    def _reset(self):
        self._state = 0
        self._episode_ended = False
        self.step_count = 0
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        self.step_count += 1
        if action == 1:
            self._state += action
        elif action == 0:
            self._state += -1
        else:
            raise ValueError('`action` should be 0 or 1.')

        if self._state<0:
            self._state=0
        elif self._state==5:
            self._episode_ended = True
        self.show_map()
        if self._episode_ended:
            return ts.termination(np.array([self._state], dtype=np.int32), 20)
        elif self.step_count >= 20:
            return ts.termination(np.array([self._state], dtype=np.int32), -20)
        else:
            reward = -1 if action == 0 else 1
            return ts.transition(np.array([self._state], dtype=np.int32), reward=reward, discount=0.7)


    def show_map(self):
        game_map = self.game_map.copy()
        game_map[self._state] = 1
        print(game_map)


