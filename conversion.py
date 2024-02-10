import numpy as np
import pprint
from functools import lru_cache
import pandas as pd

pp = pprint.PrettyPrinter(indent=4)
NUM_ROWS = 3
NUM_COLS = 4

arrs = []
cnt = 1
A = ("Up","Right","Down","Left","None")
S = [i +1 for i in range(NUM_COLS * NUM_ROWS)]
for i in range(NUM_COLS):
    arr=[]
    for j in range(NUM_ROWS):
        arr.append([cnt])
        cnt = cnt + 1
    arrs.append(np.array(arr))
grid = np.hstack(arrs)
def rc_to_s(tpl):
    return grid[tpl[0] - 1,tpl[1] - 1]

def show_layout(_arr):
    pp.pprint(np.flip(_arr.reshape(NUM_ROWS,NUM_COLS),axis=0))
def s_to_rc(val):
    for i,x in enumerate(grid):
        for j,y in enumerate(x):
            if y == val:
                return (i + 1,j + 1)
    raise RuntimeError(f"{val} Is not a valid state in s_to_rc function. Terminiating")

def is_terminal(val):
    terminal_states = [5,11,12]
    return val in terminal_states



def actions(state):
    if state in [11,12]: #Absorbing States
        return ["None"]
    if state in [5]: #Illegal State
        return ["None"]
    return ["Up","Left","Down","Right"]

@lru_cache
def calc_transition(s,action):
    if action not in A:
        raise RuntimeError(f"Invalid Action from state {s} (action = {action})")
    # if len(s) > 1:
    #     return calc_transition(s,action)
    if not isinstance(s,int):
        return np.array([calc_transition(_s,action=action) for _s in s]).reshape(-1,len(s)).T
    #IF s is terminal state return 1 at vector length and return vector
    if s == 11 or s == 12 or s == 5 or action == "None":
        P = np.zeros((NUM_ROWS,NUM_COLS))
        r,c = s_to_rc(s)
        P[r-1,c-1] = 1
        return P.flatten()

    action_to_delta = {
        "Up":(1,0),
        "Down":(-1,0),
        "Right":(0,1),
        "Left":(0,-1)
    }
    try:
        dr,dc = action_to_delta[action]
    except KeyError as e:
        raise RuntimeError(f"Invalid action mapping for action \"{action}\"")
    r,c = s_to_rc(s)

    if dr!= 0 and dc != 0:
        raise RuntimeError("You cannot not diagonally")

    P = np.zeros(shape=(NUM_ROWS,NUM_COLS))

    if dr != 0:
        new_r = r + dr
        if new_r > NUM_ROWS or new_r < 1:
            new_r = r
        if new_r == 2 and c == 2:
            new_r = r
        P[new_r -1, c -1] = 0.8
        if c < NUM_COLS and not (r == 2 and (c + 1) == 2):
            P[r - 1, c] = 0.1
        else:
            P[r-1, c -1] = P[r-1,c-1] + 0.1
        if c > 1 and not (r == 2 and (c -1) == 2):
            P[r - 1, c - 2] = 0.1
        else:
            P[r-1, c-1] = P[r-1,c-1] + 0.1

    if dc != 0:
        new_c = c  +dc
        if new_c > NUM_COLS or new_c < 1:
            new_c = c
        if r == 2 and new_c == 2:
            new_c = c
        P[r -1, new_c -1] = 0.8

        if r <NUM_ROWS and not (r+1 == 2 and c == 2):
            P[r, c-1] = 0.1
        else:
            P[r-1,c-1] = P[r-1,c-1] + 0.1
        
        if r > 1 and not (r-1 == 2 and c== 2):
            P[r-2,c-1] = 0.1
        else:
            P[r-1,c-1] = P[r-1,c-1] + 0.1
    
    return P.flatten()

show_layout(calc_transition(11,"Right")) 


# P_matricies = {a:calc_transition(S,a) for a in A} # Ask Dr. Hahsler on this one (Python vs r vectors causing lookup problems so i just lru cache the functions for speed)



@lru_cache
def P(sp,s,a):
    _mtx = calc_transition(s,a).reshape(NUM_ROWS,NUM_COLS)
    r,c = s_to_rc(sp)
    return _mtx[r-1, c-1]
    # pp.pprint(_mtx.reshape(NUM_ROWS,NUM_COLS))

def R(s,a,s_prime):
    if a == "None" or s == 11 or s == 12:
        return 0
    if s_prime == 12:
        return 1
    if s_prime == 11:
        return -1
    
    return(-0.04)

# print(P(5,4,"Up"))
print(R(12,"None",12))

pi_manual = ["Up"] * len(S)
pi_manual[10] = "Right"
pi_manual[9] = "Right"
pi_manual[8] = "Right"
pi_manual = np.array(pi_manual)
show_layout(pi_manual)

def generate_random_deterministic_policy():
    return np.array([np.random.choice(actions(s)) for a in grid for s in a]) #Return random action based on possible actions given some state s



pi_random = generate_random_deterministic_policy()
show_layout(pi_random)

def make_policy_soft(pi,epsilon=0.1):
    _pi = pi.reshape(NUM_ROWS,NUM_COLS)
    if not isinstance(pi,type(np.array([]))):
        raise RuntimeError("Policy is not a deterministic policy")
    mtx = []
    for s in S:
        _row = {a:epsilon / len(actions(s)) for a in actions(s)} 

        r,c = s_to_rc(s)
        # act_index = actions(s).index(_pi[r-1,c-1])
        _row[_pi[r-1,c-1]] = _row[_pi[r-1,c-1]] + (1- epsilon)
        mtx.append(_row)
    res = pd.DataFrame(mtx).fillna(0)
    res.index = [i + 1 for i in res.index]
    return res[['Up','Right','Down','Left','None']]
# pp.pprint(make_policy_soft(pi_random))

pi_test = np.array([
    ['Left','Right','Down',"Right"],
    ['Left','None','Up',"None"],
    ['Right','Left','Up',"None"],
]
)
def make_random_epsilon_policy(epsilon=0.1): #Ask Hahsler about this one

    mtx = []
    for s in S:
        _row = {a:np.random.rand() for a in actions(s)} 
        _row = {k:v/np.sum([_v for _v in _row.values()]) for k,v in _row.items()}
        mtx.append(_row)
    res = pd.DataFrame(mtx).fillna(0)
    res.index = [i + 1 for i in res.index]
    return res[['Up','Right','Down','Left','None']]
show_layout(pi_test)

# pp.pprint(make_policy_soft(pi_test))
# pp.pprint(make_random_epsilon_policy())
# show_layout(grid)

