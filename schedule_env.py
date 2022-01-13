from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from typing import Dict
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from itertools import product
from ScheduleReward import ScheduleReward


class ScheduleEnv(py_environment.PyEnvironment):
    # network output shape = (_observation_spec.shape[0],action count)
    def __init__(self):
        self.jobs = ['J0', 'J1', 'J2', 'J3']
        self.jobs_run_time = {'J0': 10, 'J1': 5, 'J2': 20, 'J3': 100}
        self.equipments = ['E0', 'E1']
        # action, array for jobs*equipments
        self.ac = np.array(list(product(self.jobs, self.equipments)))
        self.schedule_reward = ScheduleReward(self.jobs_run_time, self.equipments, self.ac)
        self.eqps_count = len(self.equipments)
        self.jobs_count = len(self.jobs)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.ac.shape[0] - 1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.eqps_count, self.jobs_count), dtype=np.int32, minimum=-1, maximum=self.jobs_count - 1,
            name='observation')
        self._state = np.full((self.eqps_count, self.jobs_count), -1)
        self._episode_ended = False
        self.max_step = len(self.ac)
        self.step_count = 0
        self.rewards = []
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.full((self.eqps_count, self.jobs_count), -1)
        self._episode_ended = False
        self.step_count = 0
        self.assigned_job = []
        self.rewards = []
        return ts.restart(np.array(self._state, dtype=np.int32))

    def update_state(self, empty_eqp_place, eqp_index, job_index):
        if empty_eqp_place.size != 0:
            put_job_index = empty_eqp_place[0]
            self._state[eqp_index][put_job_index] = job_index

    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        self.step_count +=1
        action_index = action
        # Make sure episodes don't go on forever.
        # action [job,eqp] this job do in this epq
        assign_job = self.ac[action_index]
        eqp_name = assign_job[1]
        job_name = assign_job[0]
        eqp_index = self.equipments.index(eqp_name)
        job_index = self.jobs.index(job_name)
        current_eqp_empty_space = np.where(self._state[eqp_index] == -1)[0]
        all_empty_space = np.where(self._state == -1)[0]
        unique_jobs = np.unique(self._state)
        assigned_job = unique_jobs[unique_jobs != -1]
        if (set(self.jobs) - set(assigned_job) == set()) or self.step_count == self.max_step or all_empty_space.size == 0:
            self._episode_ended = True

        # Agent take infinite step to take action, refine it in 100 steps
        if self._episode_ended:
            self.update_state(current_eqp_empty_space, eqp_index, job_index)
            reward = self.schedule_reward.get_episode_ended_reward(self._state)

            return ts.termination(np.array(self._state, dtype=np.int32), reward)
        else:
            reward = self.schedule_reward.get_episode_not_ended_reward(self._state, action)
            self.update_state(current_eqp_empty_space, eqp_index, job_index)
            self.rewards.append(reward)
            return ts.transition(np.array(self._state, dtype=np.int32), reward=reward, discount=1)
