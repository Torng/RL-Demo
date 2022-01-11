from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from typing import Dict
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from itertools import product
import math


class ScheduleEnv(py_environment.PyEnvironment):
    # network output shape = (_observation_spec.shape[0],action count)
    def __init__(self):
        self.jobs = ['J0', 'J1', 'J2', 'J3', 'J4', 'J5']
        self.jobs_run_time = {'J0': 10, 'J1': 5, 'J2': 20, 'J3': 12, 'J4': 100, 'J5': 10}
        self.equipments = ['E0', 'E1']
        # action, array for jobs*equipments
        self.ac = np.array(list(product(self.jobs, self.equipments)))
        self.eqps_count = len(self.equipments)
        self.jobs_count = len(self.jobs)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.ac.shape[0] - 1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.eqps_count, self.jobs_count), dtype=np.int32, minimum=-1, maximum=self.jobs_count - 1,
            name='observation')
        self._state = np.full((self.eqps_count, self.jobs_count), -1)
        self._episode_ended = False
        self.step_count = 0
        self.assigned_job = []

    def get_utilization_rate(self):
        # Two equipments for machine doing longest time
        # [first machine doing time, second machine doing time]
        eqp_run_time = [0, 0]

        # _state.shape = [eqps_count, jobs_count]
        for i, eqp in enumerate(self._state):
            for job in eqp:
                if job != -1:
                    job_name = self.jobs[job]
                    eqp_run_time[i] += self.jobs_run_time[job_name]
        max_eqp_run_time = max(eqp_run_time)
        utilization = sum([run_time / max_eqp_run_time for run_time in eqp_run_time]) / 2
        return utilization

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.full((self.eqps_count, self.jobs_count), -1)
        self._episode_ended = False
        self.step_count = 0
        self.assigned_job = []
        return ts.restart(np.array(self._state, dtype=np.int32))

    def sigmoid(self, x):
        sig = 1 / (1 + math.exp(-x))
        return sig

    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        action_index = action
        # Make sure episodes don't go on forever.
        self.step_count += 1
        # action [job,eqp] this job do in this epq
        assign_job = self.ac[action_index]
        eqp_name = assign_job[1]
        job_name = assign_job[0]
        eqp_index = self.equipments.index(eqp_name)
        job_index = self.jobs.index(job_name)
        all_empty_space = np.where(self._state[eqp_index] == -1)[0]
        job_is_exist = np.where(self._state[eqp_index] == job_index)[0]
        self.step_count += 1

        # All state are not -1
        if all_empty_space.size != 0:
            put_job_index = all_empty_space[0]
            self._state[eqp_index][put_job_index] = job_index

        if (set(self.jobs) - set(self.assigned_job) == set()) or all_empty_space.size == 0:
            self._episode_ended = True

        # raise ValueError('action' should be 0 or 1.)
        # Agent take infinite step to take action, refine it in 100 steps
        if self._episode_ended or self.step_count >= 100:
            # repeat_job
            # np.where(self._state == i) eg. np.where([[0,1,3][0,1,2]]==1) return => ([0,1],[1,1])
            repeat_job = [len(np.where(self._state == i)[0]) - 1 for i in range(self.jobs_count)]
            repeat_job = sum([j for j in repeat_job if j > 0])
            reward = self.sigmoid(self.get_utilization_rate()) * 100
            reward -= repeat_job * 100
            return ts.termination(np.array(self._state, dtype=np.int32), reward)
        else:
            reward = 0
            if job_name in self.assigned_job:
                reward = -100
            else:
                self.assigned_job.append(job_name)
                reward = 10
            return ts.transition(np.array(self._state, dtype=np.int32), reward=reward, discount=0.7)


