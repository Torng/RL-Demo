import random
import pandas as pd
import numpy as np
import time
from scipy.special import softmax



Q_TABLE = pd.DataFrame(
            np.zeros((6, 3)),  # q_table initial values
            columns=[1,-1,0],  # actions's name
        )


class Environment:
    def __init__(self, env,treasure_index):
        self.env = env
        self.location = 0
        self.treasure = treasure_index
        self.ALPHA = 1
        self.is_terminated = False
    def feedback_to_actor(self,location,action):
        reward = 0
        origin_distance = abs(self.treasure-location)
        new_location = location+action
        new_distance = abs(self.treasure-new_location)
        # if (location == 0 and action == -1):
        #     reward = -1
        if origin_distance>new_distance:
            reward = 1
        elif(new_distance==0):
            reward = 40
            self.is_terminated = True
        else:
            reward = -3
        return reward
    def show_status(self):
        print(self.env)
        if self.location == self.treasure:
            print("!!!!!!!!","re_train")
        time.sleep(0.5)
    def update_location(self,location):
        self.env[self.location] = '0'
        self.env[location] = '1'
        self.location = location
        if location == self.treasure:
            self.is_terminated = True


class Actor:
    def __init__(self, environment: Environment, actions):
        self.actions = actions
        self.environment = environment
        self.q_table = pd.DataFrame(
            np.zeros((len(self.environment.env), len(actions))),  # q_table initial values
            columns=actions,  # actions's name
        )
        self.location = 0
        self.environment.env[self.location] = '1'

    def chose_action(self):
        probability = Q_TABLE.loc[self.location,:]
        if probability.sum()<=0:
            direction = random.choice(list(probability.index))
        else:
            direction = random.choices(list(probability.index),weights=probability.values)[0]
        return direction
    def move(self,action):
        if self.location != 0 or action != -1:
            self.location += action
            self.environment.update_location(self.location)

    def update_q_table(self,action,reward):
        Q_TABLE.loc[self.location,action]+=reward
        result = softmax(Q_TABLE.loc[self.location,:])
        Q_TABLE.loc[self.location, :] = result


if __name__ == '__main__':

    actions = [1, -1, 0]
    is_terminated = False
    for i in range(100):
        print("start=============>",i)
        env = ['0', '0', '0', '0', '0', '0', '*']
        environment = Environment(env, 6)
        actor = Actor(environment, actions)
        while not environment.is_terminated:
            action = actor.chose_action()
            reward = environment.feedback_to_actor(actor.location,action)
            actor.update_q_table(action,reward)
            actor.move(action)
            environment.show_status()

    print(Q_TABLE)





