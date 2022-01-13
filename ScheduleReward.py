import numpy as np
from typing import Dict, List


class ScheduleReward:
    def __init__(self, jobs_info: Dict[str, float], equipments_info: List[str], actions):
        self.jobs = list(jobs_info.keys())
        self.jobs_run_time = jobs_info  # 先這樣
        self.eqps = equipments_info
        self.jobs_count = len(self.jobs)
        self.eqps_count = len(self.eqps)
        self.actions = actions

    def get_episode_ended_reward(self, state) -> float:
        return self.get_repeat_job_reward(state) + self.get_utilization_rate_reward(state) + self.get_remain_job_reward(
            state)

    def get_repeat_job_reward(self, state,job:int=None) -> float:
        # repeat_job
        # np.where(self._state == i) eg. np.where([[0,1,3][0,1,2]]==1) return => ([0,1],[1,1])
        if job:
            repeat_job = [len(np.where(state == job)[0]) - 1]
        else:
            repeat_job = [len(np.where(state == i)[0]) - 1 for i in range(0, self.jobs_count)]
        repeat_job = sum([j for j in repeat_job if j > 0])
        return repeat_job * -200

    def get_remain_job_reward(self, state):
        unique_jobs = np.unique(state)
        assigned_job = unique_jobs[unique_jobs != -1]
        remain_jobs = [j for j in range(len(self.jobs)) if j not in assigned_job]
        return -50 * len(remain_jobs)

    def get_utilization_rate_reward(self, state) -> float:
        eqp_run_time = [0 for _ in range(self.eqps_count)]
        assign_jobs = []
        for i, eqp in enumerate(state):
            assign_jobs = []
            for job in eqp:
                if job != -1 and job not in assign_jobs:
                    job_name = self.jobs[job]
                    eqp_run_time[i] += self.jobs_run_time[job_name]
                assign_jobs.append(job)
        max_eqp_run_time = max(eqp_run_time)
        utilization = sum([run_time / max_eqp_run_time for run_time in eqp_run_time if run_time != 0]) / 2

        return utilization * 100

    def get_episode_not_ended_reward(self, current_state, action):
        action_index = action
        # Make sure episodes don't go on forever.
        # action [job,eqp] this job do in this epq
        assign_job = self.actions[action_index]
        eqp_name = assign_job[1]
        job_name = assign_job[0]
        eqp_index = self.eqps.index(eqp_name)
        job_index = self.jobs.index(job_name)
        job_is_exist = np.where(current_state[eqp_index] == job_index)[0]
        current_eqp_empty_space = np.where(current_state[eqp_index] == -1)[0]

        if current_eqp_empty_space.size == 0:
            reward = -150
        elif job_is_exist.size == 0 and current_eqp_empty_space.size != 0:
            reward = 50
        else:
            reward = self.get_repeat_job_reward(current_state, job_index)
        return reward
