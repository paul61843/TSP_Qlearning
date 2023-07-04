import numpy as np
import random
import math

from qlearning.agent.qAgent import *

class DeliveryQAgent(QAgent):

    def __init__(self,*args,**kwargs):
        super().__init__(**kwargs)
        self.reset_memory()

    def act(self,s):

        # Get Q Vector
        q = np.copy(self.Q[s,:])

        # Avoid already visited states
        q[self.states_memory] = -np.inf

        if np.random.rand() > self.epsilon:
            # tau = 1.0  # 溫度參數
            # probabilities = np.exp(q / tau) / np.sum(np.exp(q / tau))

            # if np.isnan(q).any():
            #     # 使用Boltzmann Exploration選擇行動
            #     a = np.random.choice(range(len(q)), p=probabilities )
            # else: 
                a = np.argmax(q)
        else:
            a = np.random.choice([x for x in range(self.actions_size) if x not in self.states_memory])

        return a


    def remember_state(self,s):
        self.states_memory.append(s)

    def reset_memory(self):
        self.states_memory = []
