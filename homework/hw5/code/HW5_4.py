import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

GOAL = 4 # upper right corner
START = 20 # lower left corner
SNAKE1 = 7
SNAKE2 = 17

eps = 0.25

class Robot_vs_snakes_world(discrete.DiscreteEnv):
    def __init__(self):
        self.shape = [5, 5]
        
        nS = np.prod(self.shape) # total states
        nA = 4 # total actions
        
        MAX_X, MAX_Y = self.shape
        
        P = {}
        grid = np.arange(nS).reshape(self.shape)
        """
        grid:  
               [ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24]
        """
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            
            P[s] = {a: [] for a in range(nA)}
            """{0: [], 1: [], 2: [], 3: []}"""
            is_done = lambda s: s == GOAL # location
            
            if is_done(s):
                reward = 0. # OK.
            elif s in [SNAKE1, SNAKE2]:
                reward = -15. # OUCH
            else:
                reward = -1. # BLEEDING
                
            if is_done(s):
                P[s][UP]=[(1.0,s,reward,True)]
                P[s][RIGHT]=[(1.0,s,reward,True)]
                P[s][DOWN]=[(1.0,s,reward,True)]
                P[s][LEFT]=[(1.0,s,reward,True)]
            else:
                # where to move (0 if cant move, else move) 
                # (since s in [0, 1 .., 24]
                ns_up = s if y==0 else s-MAX_X
                ns_right = s if x==(MAX_X-1) else s+1
                ns_down = s if y==(MAX_Y-1) else s+MAX_X
                ns_left = s if x==0 else s-1
                
                # prob, next state , reward, boolean finish
                # reward in [0, -15, -1]
                P[s][UP] = [(1-(2*eps),ns_up,reward,is_done(ns_up)),
                            (eps,ns_right,reward,is_done(ns_right)),
                            (eps,ns_left,reward,is_done(ns_left))]
                
                P[s][RIGHT]=[(1-(2*eps),ns_right,reward,is_done(ns_right)),  
                             (eps,ns_up,reward,is_done(ns_up)),
                             (eps,ns_down,reward,is_done(ns_down))]
                
                P[s][DOWN] = [(1-(2*eps),ns_down,reward,is_done(ns_down)),
                              (eps,ns_right,reward,is_done(ns_right)),
                              (eps,ns_left,reward,is_done(ns_left))]
                
                P[s][LEFT] = [(1-(2*eps) , ns_left , reward , is_done(ns_left)) ,
                              ( eps , ns_up , reward , is_done(ns_up)) ,
                              ( eps , ns_down , reward , is_done(ns_down))]
                
            it.iternext()
            
        isd = np.zeros(nS)
        isd[START] = 1.
        self.P = P # {}
        
        super(Robot_vs_snakes_world, self).__init__(nS , nA , P , isd)
                
    def _render(self):
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            
            if self.s == s:
                output = u" R "
            elif s == GOAL:
                output = u" G "
            elif s in [SNAKE1, SNAKE2]:
                output =  u" S "
            else:
                output = u" o "
                
            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()
                
            sys.stdout.write(output)
            
            if x == self.shape[1] - 1:
                sys.stdout.write("\n")
                
            it.iternext()
            
        sys.stdout.write("\n")
        
def bellman_equation(state, V):
    state_values = []
    for action in range(env.nA):
        next_info = env.P[state][action]
        components = []
        for next_prob, next_state, reward, _ in next_info:
            components.append(next_prob * (reward + discount_factor * V[next_state]))
        state_values.append(sum(components))
    return max(state_values)

def deterministic_policy(V, policy):
    for state in range(env.nS):
        state_values = []
        for action in range(env.nA):
            next_info = env.P[state][action]
            components = []
            for next_prob, next_state, reward, _ in next_info:
                components.append(next_prob * (reward + discount_factor * V[next_state]))
            state_values.append(sum(components))
        a = np.argmax(state_values)
        policy[state, a] = 1
    return policy

def value_iteration(env):    
    # initialise V(s) for all s
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for state in range(env.nS):
            v_prev = V[state]
            V[state] = bellman_equation(state, V)
            #update delta
            delta = max(delta, np.abs(v_prev - V[state]))
        if delta<theta:
            break
            
    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    policy = deterministic_policy(V, policy)
    return policy, V

if __name__ == "__main__":
    theta = 0.0000001 # small algorithm parameter
    discount_factor = 1.
    
    env = Robot_vs_snakes_world()
    policy, V = value_iteration(env)

    env._render()
    while env.s != GOAL:
        DIR = np.argmax(policy[env.s])
        env.step(DIR)
        env._render()

    print("Policy:")
    print(policy)

    
