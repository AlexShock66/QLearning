import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

NUM_ROWS = 3
NUM_COLS = 4

arrs = []
cnt = 1
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
    pp.pprint(np.flip(_arr,axis=0))
def s_to_rc(val):
    for i,x in enumerate(grid):
        for j,y in enumerate(x):
            if y == val:
                return (i + 1,j + 1)

def is_terminal(val):
    terminal_states = [5,11,12]
    return val in terminal_states



def actions(state):
    if state in [11,12]: #Absorbing States
        return ["None"]
    if state in [5]: #Illegal State
        return ["None"]
    return ["Up","Left","Down","Right"]


def calc_transition(s,actions):
    if len(s) > 1:
        return calc_transition(s,actions)
    

# print(s_to_rc(12))
# print(rc_to_s((3,4)))



