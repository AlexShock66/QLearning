import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np
from tqdm.auto import tqdm
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from enum import Enum
import pickle
from multiprocessing import pool
import math
from sklearn.model_selection import train_test_split
import os
import datetime
import json

class TransformerBlock(tf.keras.layers.Layer): # inherit from Keras Layer
    """Custom transformer block class to use in the model creation process if needed
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2):
        super().__init__()
        # setup the model heads and feedforward network
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embed_dim)

        # make a two layer network that processes the attention
        self.ffn = tf.keras.Sequential()
        self.ffn.add( tf.keras.layers.Dense(ff_dim, activation='tanh') )
        self.ffn.add( tf.keras.layers.Dense(embed_dim) )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-3)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-3)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        # apply the layers as needed (similar to PyTorch)

        # get the attention output from multi heads
        # Using same inpout here is self-attention
        # call inputs are (query, value, key)
        # if only two inputs given, value and key are assumed the same
        attn_output = self.att(inputs, inputs)

        # create residual output, with attention
        out1 = self.layernorm1(inputs + attn_output)

        # apply dropout if training
        out1 = self.dropout1(out1, training=training)

        # place through feed forward after layer norm
        ffn_output = self.ffn(out1)
        # print("asujdhfasdf",ffn_output.shape)
        out2 = self.layernorm2(out1 + ffn_output)

        # apply dropout if training
        out2 = self.dropout2(out2, training=training)
        #return the residual from Dense layer
        return out2

def generate_custom_kernels():
    """Generates custom 4x4 kernels that describe the possible win conditions of a connect 4 board. Returns biases,weights for said conditions where bias terms are all 0

    Returns:
        list[np.ndarray]: List of [bias,weights] to use in tensorflow `set_weights` function for convolutional layer.
    """
    synthetic_weights= np.array([
    [
        [0,0,0,0],
        [1,1,1,1],
        [0,0,0,0],
        [0,0,0,0]
    ],
    [
        [1,1,1,1],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0]
    ],
    [
        [0,0,0,0],
        [0,0,0,0],
        [1,1,1,1],
        [0,0,0,0]
    ],
    [
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [1,1,1,1]
    ],
    [
        [1,0,0,0],
        [1,0,0,0],
        [1,0,0,0],
        [1,0,0,0]
    ],
    [
        [0,1,0,0],
        [0,1,0,0],
        [0,1,0,0],
        [0,1,0,0]
    ],
    [
        [0,0,1,0],
        [0,0,1,0],
        [0,0,1,0],
        [0,0,1,0]
    ],
    [
        [0,0,0,1],
        [0,0,0,1],
        [0,0,0,1],
        [0,0,0,1]
    ],
    [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ],
    [
        [0,0,0,1],
        [0,0,1,0],
        [0,1,0,0],
        [1,0,0,0]
    ]
    ])
    synthetic_weights = np.concatenate([arr.reshape(4,4,1,1) for arr in synthetic_weights],axis=3)

    return [np.array([0.0] * synthetic_weights.shape[-1]), synthetic_weights] #return bias, kernel weights


def custom_sigmoid(x):
    return (((1 / (1 + math.e**-x)) * 2) -1)


def check_board_win(board):
    """Checks to see if a board has reached a termnial state with convolutions of possible win conditions.

    Args:
        board (np.ndarray): The board for which to check the conditions on. Needs to be 2d representation of nrows by ncols. Must contain only -1,1, and 0 for player 2, player1, and empty cells respectivly

    Returns:
        Int || None: Will return 1 if player 1 wins, -1 if player -1 wins, 0 if game is still going and None if the game results in a tie
    """
    horizontal_kernel = np.array([[ 1, 1, 1, 1]])
    vertical_kernel = np.transpose(horizontal_kernel)
    diag1_kernel = np.eye(4, dtype=np.uint8)
    diag2_kernel = np.fliplr(diag1_kernel)
    detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
    for kernel in detection_kernels:

        a = convolve2d(board,kernel,mode='valid')
        if( (a == 4).any()):
            return 1
        if ((a == -4).any()):
            return -1
        
        # print(a.any())
    if(len([i for i,j in enumerate(board[0]) if j == 0]) == 0): return None
    return 0
def evaluate_board(board,player):
    """Heuristic function for evaluating non terminal board states. This function prioritizes getting unblocked pairs, triplets, and 7 traps. Higher return values are desirable. The result of this function is bounded between -1 and 1

    Args:
        board (np.ndarray): Board representation in nrows by ncols with -1,1,0 as values for player2, player1, and empty cell respectivly
        player (int): Either -1 or 1. Will determine if we are looking at the board as player 1 or as player -1. Mainly will just affect the sign of the return value based on which player's turn it is

    Returns:
        float: The heuristic value for the board state. Bounded between -1 and 1. NOTE: can be bounded by anything you like, but the rewards are bounded by -1 and 1 so it can cause unexpected results if the value is outside of these bounds for the training session
    """
    _possible_winner = check_board_win(board)
    if _possible_winner == player:
        return 1
    elif _possible_winner == player * -1:
        return -1
    
    horizontal_kernel = np.array([[ 1, 1, 1, 1]])
    vertical_kernel = np.transpose(horizontal_kernel)
    diag1_kernel = np.eye(4, dtype=np.uint8)
    diag2_kernel = np.fliplr(diag1_kernel)
    detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
    utils = []
    for kernel in detection_kernels:
        a = convolve2d(board,kernel,mode='valid')
        if(a == (3 * player)).any():
            times = np.count_nonzero(a == (3 * player))
            utils += [0.75] * times
        if(a == (2 * player)).any():
            times = np.count_nonzero(a == (2 * player))
            utils += [0.1] * times
        if(a == (3 * player) * -1).any():
            times = np.count_nonzero(a == (3 * player * -1))
            utils += [-0.75] * times
        if(a == (2 * player) * -1).any():
            times = np.count_nonzero(a == (2 * player * -1))
            utils += [-0.1] * times
    seven_trap = np.array([
        [1,1,1],
        [0,1,0],
        [1,0,0]
    ])
    seven_kernels = [seven_trap]
    seven_kernels.append(np.array([
        [1,1,1],
        [0,1,0],
        [0,0,1]
    ]))
    seven_kernels = [np.flip(arr) for arr in seven_kernels]
    for kernel in seven_kernels:
        a = convolve2d(board,kernel,mode='valid')
        if(a == (5 * player)).any():
            times = np.count_nonzero(a == (5 * player))
            utils += [0.85] * times
        if(a == (5 * player * -1)).any():
            times = np.count_nonzero(a == (5 * player * -1))
            utils += [-0.85] * times

    if len(utils) != 0:
        return custom_sigmoid(np.sum(utils))
    else: return 0

def visualize(board):
    """Helper function that will visualize a nrows by ncols board using maplotlib

    Args:
        board (np.ndarray): Board representation in nrows by ncols with -1,1,0 as values for player2, player1, and empty cell respectivly
    """
    plt.axes()
    rectangle=plt.Rectangle((-0.5,len(board)*-1+0.5),len(board[0]),len(board),fc='blue')
    circles=[]
    for i,row in enumerate(board):
        for j,val in enumerate(row):
            color='white' if val==0 else 'red' if val==1 else 'yellow'
            circles.append(plt.Circle((j,i*-1),0.4,fc=color))

    plt.gca().add_patch(rectangle)
    for circle in circles:
        plt.gca().add_patch(circle)

    plt.axis('scaled')
    plt.show()

def play(board:np.ndarray,p1,p2):
    """Plays a game to completion starting from some board state and using player policies p1 and p2

    Args:
        board (np.ndarray): _description_
        p1 (BasePlayer): Player 1 in the game to play to completion   
        p2 (BasePlayer): Player 2 in the game to play to completion

    Returns:
        int | None: player symbol if game is terminal, or None if game results in a tie
    """
    b = Board(p1=p1,p2=p2,num_rows=len(board[0]),num_cols=len(board))
    b.reset()
    b.board = board
    return b.play_agents(verbose=0)

def get_expected_value(board,player_symbol,num_iterations=500,win_reward = 1.0,tie_reward=0.5,lose_reward=-1.0):
    """Secondary way of getting a heuristic for a board. This method will use monte carlo search to determine the reward to give at any non terminal state but will take MUCH longer to run than `evaluate_board()`

    Args:
        board (np.ndarray): board representation in nrows by ncols
        player_symbol (int): Player symbol for who we are evaluating the board for (either -1 or 1)
        num_iterations (int, optional): How many iterations to simulate with monte carlo search. Defaults to 500.
        win_reward (float, optional): Reward for winning the game. Defaults to 1.0.
        tie_reward (float, optional): Reward for tieing in the game. Defaults to 0.5.
        lose_reward (float, optional): Reward for losing the game. Defaults to -1.0.

    Returns:
        float: Heuristic value for the board state that is the expected value of the game based on the rewards given.
    """
    player_vals,player_counts = np.unique(board,return_counts=True)
    tmp_dct = {k:v for k,v in zip(player_vals,player_counts)}
    red_pieces = tmp_dct.get(1,0)
    yellow_pieces = tmp_dct.get(-1,0)
    if red_pieces == yellow_pieces:
        p1 = RandomPlayer(player_symbol=1)
        p2 = RandomPlayer(player_symbol=-1)
    else:
        p1 = RandomPlayer(player_symbol=-1)
        p2 = RandomPlayer(player_symbol=1)
    func_args = [(board.copy(),p1,p2) for i in range(num_iterations)]
    with pool.Pool() as p:
        winners = p.starmap(play,func_args)
        vals,counts = np.unique([v if v is not None else 0 for v in winners],return_counts=True)
        v = {k:float(v/num_iterations) for k,v in zip(vals,counts)}
    return win_reward * v.get(player_symbol,0.0) + tie_reward * v.get(0,0.0) + lose_reward * v.get(player_symbol * -1,0.0)
 
def playout(board:np.ndarray,num_iterations=300):
    """Simulates a game to completion given some board state. Will use num_iterations games of random play to determine the winrates, loserates, and tie rates.

    Args:
        board (np.ndarray): board state in numpy representation
        num_iterations (int, optional): Number of games to simulate with the random players. Defaults to 300.

    Returns:
        dict[int:float]: Dictionary keyed by game result (-1 for -1player wins, 1 for p1 wins, 0 for tie) with the value being the percentage of the time that the result occurred
    """
    player_vals,player_counts = np.unique(board,return_counts=True)
    tmp_dct = {k:v for k,v in zip(player_vals,player_counts)}
    red_pieces = tmp_dct.get(1,0)
    yellow_pieces = tmp_dct.get(-1,0)
    if red_pieces == yellow_pieces:
        p1 = RandomPlayer(player_symbol=1)
        p2 = RandomPlayer(player_symbol=-1)
    else:
        p1 = RandomPlayer(player_symbol=-1)
        p2 = RandomPlayer(player_symbol=1)
    
    b = Board(p1=p1,p2=p2,num_rows=len(board[0]),num_cols=len(board))
    winners = []
    for i in range(num_iterations):
        b.reset()
        b.board = board.copy()
        winners.append(b.play_agents(verbose=0))
    vals,counts = np.unique([v if v is not None else 0 for v in winners],return_counts=True)
    return {k:float(v/num_iterations) for k,v in zip(vals,counts)}


class BasePlayer:
    """BasePlayer class to inherit all other players for. Should call this constructor in inherited constructors in order to ensure that all required values exist, even if not using them
    """
    def __init__(self,player_symbol) -> None:
        self.player_symbol = player_symbol
        self.num_rows = 0
        self.num_cols = 0
        self.epsilon = 0.0
        self.initial_epsilon = 0.0
    def choose_action(self,valid_actions=None,board=None): #Function signature for choose_action. Called to get the desired column for the agent to place a piece in
        raise NotImplementedError("Player class is virtual, please inherent from it")
    def give_reward(self,reward,s,s_prime,action): #Function signature for give_reward. Called after each move to feed the agent some reward as a float value and inform it of s, s_prime, and the action that it took to get said reward.
        raise NotImplementedError("Player class is virtual, please inherent from it")
    def reset(self): #Function signature for reset. Is called after every iteration if not a DeepQ player, and is called every train_iterations if is DeepQ player
        raise NotImplementedError("Player class is virtual, please inherent from it")

class MonteCarloPlayer(BasePlayer):
    """
    Class that uses a monte carlo simulation of different possible actions and picks the action with the highest winrate after n simulations per action
    """
    def __init__(self, player_symbol,num_sims=500,win_reward=1.0,tie_reward=0.5,lose_reward=-1.0) -> None:
        super().__init__(player_symbol)
        self.num_sims = num_sims
        self.win_reward = win_reward
        self.tie_reward = tie_reward
        self.lose_reward = lose_reward
    def give_reward(self, reward, s, s_prime, action):
        pass
    def reset(self):
        pass
    def choose_action(self, valid_actions=None, board=None):
        func_args = [(get_next_board_state(board,a,self.player_symbol),self.num_sims) for a in valid_actions]
        with pool.Pool(processes=self.num_cols) as p:
            results = p.starmap(playout,func_args)
            expected_values = [self.win_reward * v.get(self.player_symbol,0.0) + self.tie_reward * v.get(0,0.0) + self.lose_reward * v.get(self.player_symbol * -1,0.0) for v in results]

            return valid_actions[expected_values.index(max(expected_values))]

class InputRemapper(tf.keras.layers.Layer):
    """
    Custom tensorflow layer that reshapes board input of shape (None,nrows,ncols,1) into a 3 channel tensor of shape (None,nrows,ncols,3) where each channel is a binary 2d matrix for a given player or for empty cells. (The channels are in the following order: player 1,player -1,empty cells)
    """
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
    def call(self, inputs, *args, **kwargs):
        return tf.cast(tf.concat([inputs == 1, inputs == -1, inputs == 0],axis=len(inputs.shape) - 1),tf.float32)

class RandomPlayer(BasePlayer):
    """
    Player who will randomly play an action. Does not make any informed decisions.
    """
    def __init__(self, player_symbol) -> None:
        super().__init__(player_symbol)
    def choose_action(self, valid_actions=None, board=None):
        return np.random.choice(valid_actions)
    def give_reward(self, reward,s,s_prime,action):
        pass
    def reset(self):
        pass

def get_next_board_state(board,action,player_symbol):
    """Will get the next representation of the board given some action and some player symbol and the current board.

    Args:
        board (np.ndarray): Numpy representation of the board in nrows x numcols.
        action (int): Action that should be added to the board to get state prime
        player_symbol (int): Player symbol to place on the board

    Returns:
        np.ndarray | None: Will return the new board state if it is valid, otherwise will return None if there was an issue placing the piece in the given column
    """
    try:
        _board = board.copy()
        col = _board[:,action]
        idx = int(np.where(col == 0)[0][-1])
        col[idx] = player_symbol
        return _board
    except:
        return None
def get_possible_moves(board):
    """Returns all possible moves given some board state

    Args:
        board (np.ndarray): Board represented as an numpy array of nrows by ncols 

    Returns:
        list[int]: A list of all possible actions from the given board state
    """
    if check_board_win(board) == 0:
        return [i for i,j in enumerate(board[0]) if j == 0]
    return []


class ACTION_REASONING(Enum): # Used to see why the agent will make a move for DeepQ. Not necessary but this information is stored in the QPlayers regardless.
    RANDOM = 1
    POLICY = 2

class QPlayer(BasePlayer):
    def __init__(self, player_symbol,epsilon = 0.5,gamma=0.5,alpha=0.4,name='Q_player') -> None:
        super().__init__(player_symbol)
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.action_history = []
        self.action_reasoning = []
        self.board_history = []
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}
        self.name = name
    def show_q_function(self,state):
        q_vals = self.q_table.get(self.get_board_hash(state),None)
        if q_vals is None:
            print("No Q Values found for state")
            return [-999.0] * self.num_cols
        return q_vals
    def get_board_hash(self,board:np.ndarray):
        # return str(board.flatten())
        return board.tobytes()
    def savePolicy(self):
        with open(os.path.join('QTables/',f'{datetime.datetime.now().strftime("%m_%d_%Y:%H:%M")}_{str(self.name)}')) as fw:
            pickle.dump(self.q_table, fw)
        

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.q_table = pickle.load(fr)
        fr.close()
    def choose_action(self, valid_actions=None, board=None):
        action = None
        
        if np.random.random() < self.epsilon:
            action = np.random.choice(valid_actions)
            self.action_history.append(action)
            self.action_reasoning.append(ACTION_REASONING.RANDOM)
            self.board_history.append(board.copy())
        else:
            st_reward = self.q_table.get(self.get_board_hash(board),None)
            if st_reward is not None:
                st_reward = np.array([v if i in valid_actions else - 999.0 for i,v in  enumerate(st_reward)]) #Force agent to not make invalid actions
                tmp = max(st_reward)

                action = np.random.choice(np.where(st_reward == max(st_reward))[0]) #Choose a random action from all actions that have the highest q values
                self.action_history.append(action)
                self.action_reasoning.append(ACTION_REASONING.POLICY)
                self.board_history.append(board.copy())

            else:
                action = np.random.choice(valid_actions)
                self.action_history.append(action)
                self.action_reasoning.append(ACTION_REASONING.POLICY)
                self.board_history.append(board.copy())

        return action
    def reset(self):
        self.epsilon = self.initial_epsilon
        self.action_history.clear()
        self.action_reasoning.clear()
        self.board_history.clear()
    
    def get_transition_probabilities(self,board,how='random'):
        if how.lower() == 'rand' or how.lower() == 'random':
            mvs = get_possible_moves(board)
            return [(get_next_board_state(board,m,self.player_symbol * -1),1.0/len(mvs)) for m in mvs]
        elif how.lower() == 'q':
            raise NotImplementedError("Did not implement q transition probs yet")


    def give_reward(self, reward,s,s_prime,action):
        q_s_a = self.q_table.get(self.get_board_hash(s),None)
        q_s_prime = self.q_table.get(self.get_board_hash(s_prime),None)
        if q_s_a is None:
            q_s_a = np.array([0.0] * len(s[0]),dtype=float)
            self.q_table[self.get_board_hash(s)] = q_s_a.copy()

        if q_s_prime is None:
            q_s_prime = np.array([0.0] * len(s[0]),dtype=float)
            self.q_table[self.get_board_hash(s_prime)] = q_s_prime.copy()
            
        max_action = max(q_s_prime)
        new_q = (1-self.alpha) * q_s_a[action] + self.alpha * (reward + self.gamma * max_action)
        q_s_a[action] = new_q
        self.q_table[self.get_board_hash(s)] = q_s_a.copy()
        

class GreedyPlayer(BasePlayer):
    def __init__(self, player_symbol) -> None:
        super().__init__(player_symbol)
    def give_reward(self, reward, s, s_prime, action):
        pass
    def reset(self):
        pass
    def choose_action(self, valid_actions=None, board=None):
        win_move = None
        stop_lose_move = None
        for a in valid_actions:
            next_board = get_next_board_state(board.copy(),a,self.player_symbol)
            if check_board_win(next_board) == self.player_symbol:
                win_move = a
            elif check_board_win(get_next_board_state(board.copy(),a,-1 * self.player_symbol)) == -1 * self.player_symbol:
                stop_lose_move = a
        
        if win_move is not None:
            return win_move
        if stop_lose_move is not None:
            return stop_lose_move
        
        return np.random.choice(valid_actions)


class DeepQPlayer(QPlayer):
    def __init__(self, player_symbol, epsilon=0.5, gamma=0.5, alpha=0.4, name='Q_player',network=None,training_batch_size=256,max_eval_batch_size=4096) -> None:
        super().__init__(player_symbol, epsilon, gamma, alpha, name)
        # self.q_network = network.
        self.max_eval_batch_size = max_eval_batch_size

        if network is not None:
            self.q_network = tf.keras.models.clone_model(network)
            self.q_prime_network = tf.keras.models.clone_model(network)
        else:
            self.q_network = None
            self.q_prime_network=None
        self.reward_history = []
        self.s_prime_history = []
        self.train_history = None
        self.training_batch_size = training_batch_size        
    def copy_model_over(self):
        self.q_prime_network.set_weights(self.q_network.get_weights())
        self.q_prime_network.trainable = False
        for l in self.q_prime_network.layers:
            l.trainable = False


    def show_q_function(self, state):
        return np.array([self.q_network([state.reshape(1,self.num_rows,self.num_cols,1),np.array(self.convert_action_to_1_hot(a)).reshape(-1,self.num_cols)]) for a in range(self.num_cols)]).flatten()


    def show_target_values(self,state,alpha,opponent=GreedyPlayer(player_symbol=-1)):

        states = [get_next_board_state(state,action=i,player_symbol=self.player_symbol) if i in get_possible_moves(state) else state for i in range(self.num_cols)]

        possible_utils = []
        for s in states:
            if check_board_win(s) == self.player_symbol:
                possible_utils.append(1.0)
            else: possible_utils.append(0.0)

        s_prime_history = [get_next_board_state(s,action=opponent.choose_action(get_possible_moves(s),s),player_symbol=self.player_symbol * -1) if len(get_possible_moves(s)) != 0 else s for s in states]

        a_primes = []
        s_primes = []
        for s in s_prime_history:
            for i in range(self.num_cols):
                a_primes.append(self.convert_action_to_1_hot(i))
                s_primes.append(s.copy())
        

        
        a_primes = np.array(a_primes)
        state_primes = np.array(s_primes).reshape(-1,self.num_rows,self.num_cols,1)
        q_star_values = self.q_prime_network([state_primes,a_primes]).numpy().reshape(-1,self.num_cols)

        max_q_values = []
        for possible_q_vals,sp in zip(q_star_values,s_prime_history):
            max_q_values.append(max([v if i in get_possible_moves(sp.reshape(self.num_rows,self.num_cols)) else -999.0 for i,v in enumerate(possible_q_vals)]))
            
        max_q_values = [i if i != -999.0 else 0 for i in max_q_values]
        target = [(alpha)*(q * self.gamma) + (1 - alpha) * r if (r != 1 and r != -1) else r for q,r in zip(max_q_values,[evaluate_board(s,self.player_symbol) for s in s_prime_history])]


        return target
        
        

    def generate_network(self):
        board_input = tf.keras.layers.Input(shape=(self.num_rows,self.num_cols,1))
        action_input = tf.keras.layers.Input(shape=self.num_cols)

        flat_board = tf.keras.layers.Flatten()(board_input)

        combined_input = tf.keras.layers.concatenate([flat_board,action_input])

        main_path = tf.keras.layers.Dense(256)(combined_input)

        main_path = tf.keras.layers.Dropout(0.4)(main_path)
        main_path = tf.keras.layers.concatenate([main_path,combined_input])
        main_path = tf.keras.layers.Dense(128)(main_path)
        main_path = tf.keras.layers.Dropout(0.5)(main_path)
        main_path = tf.keras.layers.Dense(1,activation='tanh')(main_path)

        model = tf.keras.Model(inputs=[board_input,action_input],outputs=main_path)
        model.compile(
            loss = tf.keras.losses.MeanSquaredError(),
            optimizer = tf.keras.optimizers.Adam(),
            metrics=['cosine_similarity']
        )
        return model

    def convert_action_to_1_hot(self,action):
        _arr = np.array([0] * self.num_cols)
        _arr [action] = 1
        return _arr
    
    
    def choose_action(self,valid_actions=None,board:np.ndarray=None,save_choices=False):  
        if self.q_network is None or self.q_prime_network is None:
            self.q_network = self.generate_network()
            self.q_prime_network = self.generate_network()
            self.copy_model_over() #NOTE: Should I do this? or leave them randomly initialized seperately from eachother
        
        if np.random.random() < self.epsilon:
            action = np.random.choice(valid_actions)
            if save_choices:
                self.board_history.append(board.copy())
                self.action_history.append(action)
                self.action_reasoning.append(ACTION_REASONING.RANDOM)
            return action
        
        
        oh_actions = np.row_stack([self.convert_action_to_1_hot(a) for a in range(self.num_cols)])
        boards = np.row_stack([board.reshape(1,self.num_rows,self.num_cols,1) for i in range(self.num_cols)])
        # print(f'OH: {oh_actions.shape}\nbrds:{boards.shape}')
        q_values = self.q_network([boards,oh_actions]).numpy().flatten()

        q_values = [q if i in valid_actions else -999.0 for i,q in enumerate(q_values)] #Force it to take a valid action
        action = np.random.choice(np.where(q_values == max(q_values))[0])

        if save_choices:
            self.action_history.append(action)
            self.board_history.append(board.copy())
            self.action_reasoning.append(ACTION_REASONING.POLICY)

        return action
        
    def give_reward(self,reward,s=None,s_prime=None,action=None,n_train_epochs=3):
        self.s_prime_history.append(s_prime.copy())
        self.reward_history.append(reward)
        self.action_history.append(action)
        self.board_history.append(s.copy())
    

    def augment_train_data(self,states:np.ndarray,actions:np.ndarray,targets:list):
        _states = []
        _actions = []
        _targets = []
        num_augmented_states = 0
        for s,a,t in zip(states.reshape(-1,self.num_rows,self.num_cols),actions,targets):
            _states.append(s.copy())
            _actions.append(a)
            _targets.append(t)
            invalid_actions = [i for i in range(self.num_cols) if i not in get_possible_moves(s)]
            for ia in invalid_actions:
                _states.append(s.copy())
                _actions.append(self.convert_action_to_1_hot(ia))
                _targets.append(0.0)
                num_augmented_states += 1
        print(f'Number of states augmented: {num_augmented_states}')
        
        return np.array(_states).reshape(-1,self.num_rows,self.num_cols,1),np.array(_actions),_targets
    def train_network(self,epochs=5):
        states = np.array(self.board_history).reshape(-1,self.num_rows,self.num_cols,1)
        actions = np.row_stack([self.convert_action_to_1_hot(a) for a in self.action_history])
        
        a_primes = []
        s_primes = []
        for s in self.s_prime_history:
            for i in range(self.num_cols):
                a_primes.append(self.convert_action_to_1_hot(i))
                s_primes.append(s.copy())
        

        
        a_primes = np.array(a_primes)
        state_primes = np.array(s_primes).reshape(-1,self.num_rows,self.num_cols,1)


        q_star_values = self.q_prime_network([state_primes,a_primes]).numpy().reshape(-1,self.num_cols)
        
        max_q_values = []
        for possible_q_vals,sp in zip(q_star_values,self.s_prime_history):
            max_q_values.append(max([v if i in get_possible_moves(sp.reshape(self.num_rows,self.num_cols)) else -999.0 for i,v in enumerate(possible_q_vals)]))
            
        max_q_values = [i if i != -999.0 else 0 for i in max_q_values]
        target = [(self.alpha)*(q * self.gamma) + (1 - self.alpha) * r if (r != 1 and r != -1) else r for q,r in zip(max_q_values,self.reward_history)]

        
        targets = target
        print(f""" Before Random Replay:
            States Shape: {states.shape}
            Actions Shape: {actions.shape}
            Len targets: {len(targets)}
            """)
        targets = np.array(targets).reshape(-1,1)
        print(f'Max Target: {max(targets.flatten())}')
        print(f'Min Target: {min(targets.flatten())}')
        print(f'Std Targets: {np.std(targets)}')


        train_states,test_states,train_actions,test_actions,train_targets,test_targets = train_test_split(states,actions,targets)

        train_hist = self.q_network.fit(x=[train_states,train_actions],y=train_targets,epochs=epochs,batch_size=self.training_batch_size, validation_data=([test_states,test_actions],test_targets))

        new_dct:dict= train_hist.history
        if self.train_history is None:
            self.train_history = new_dct
        else:
            for k,v in new_dct.items():
                for new_value in v:
                    self.train_history[k].append(new_value)

    def savePolicy(self):
        date_str = datetime.datetime.now().strftime("%m_%d_%Y:%H:%M")

        self.q_network.save(os.path.join('NewQNetworks/',f'{date_str}_{self.name}'))

        with open(os.path.join('NewHistories/',f'{date_str}_{self.name}_history.json'),'w') as fp:
            json.dump(self.train_history,fp)
        print(f'Policy Saved for {self.name}')

    def loadPolicy(self, file):
        self.q_network = tf.keras.models.load_model(file)
        self.q_prime_network = tf.keras.models.load_model(file)
        print(f'{self.name} Loaded Q and Q Prime networks from file {file}')
    def reset(self):
        self._clear_memory()
        pass
    def _clear_memory(self):
        self.action_reasoning.clear()
        self.board_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        self.s_prime_history.clear()

class SinglePredictionQPlayer(DeepQPlayer):
    def __init__(self, player_symbol, epsilon=0.5, gamma=0.5, alpha=0.4, name='Q_player', network=None,train_batch_size=8) -> None:
        super().__init__(player_symbol, epsilon, gamma, alpha, name, network,training_batch_size=train_batch_size)
        


    def convert_action_to_1_hot(self, action):
        return [action]
    def generate_network(self):
        board_input = tf.keras.layers.Input(shape=(self.num_rows,self.num_cols,1))
        action_input = tf.keras.layers.Input(shape=(1))
        
        # board_flat = InputRemapper()(board_input)
        # board_flat = tf.keras.layers.Flatten()(board_flat)

        # action_flat = tf.keras.layers.Flatten()(action_input)

        # board_path = tf.keras.layers.concatenate([board_flat,action_flat])
        # board_path = tf.keras.layers.Dense(128,activation='relu')(board_path)

        # board_path = tf.keras.layers.concatenate([board_path,action_flat,board_flat])
        # board_path = tf.keras.layers.Dropout(0.5)(board_path)

        # board_path = tf.keras.layers.Dense(64,activation='relu')(board_path)
        # board_path = tf.keras.layers.concatenate([board_path,action_flat,board_flat])
        # board_path = tf.keras.layers.Dropout(0.3)(board_path)
        
        # board_path = tf.keras.layers.Dense(32,activation='relu')(board_path)
        # board_path = tf.keras.layers.Dropout(0.35)(board_path)
        # joined = tf.keras.layers.Dense(1,activation='tanh')(board_path)
        ################################################## Pre 4_18 ##################################################
        board_path = InputRemapper()(board_input)
        board_path = tf.keras.layers.Conv2D(filters=32,kernel_size=(4,4),padding='same')(board_path)
        board_path = tf.keras.layers.Flatten()(board_path)
        board_path = tf.keras.layers.Dense(64)(board_path)
        embed = tf.keras.layers.Embedding(input_dim=self.num_cols,output_dim=64)(action_input)

        joined = tf.keras.layers.multiply([embed,board_path])

        joined = tf.keras.layers.Dropout(0.25)(joined)

        joined = tf.keras.layers.Dense(64)(joined)

        joined = tf.keras.layers.Dense(1,activation='tanh')(joined)
        ################################################## Pre 4_18 ##################################################
        model = tf.keras.Model(inputs = [board_input,action_input],outputs=joined)
        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(),
            metrics =['cosine_similarity']
        )
        return model

    
    def show_q_function(self, state):
        states = []
        [states.append(state.copy()) for s in range(self.num_cols)]
        actions = np.arange(start=0,stop=self.num_cols).reshape(-1,1)
        states = np.array(states).reshape(-1,self.num_rows,self.num_cols,1)
        return self.q_network([states,actions]).numpy().flatten()

class MultiPredictionQPlayer(SinglePredictionQPlayer):
    def __init__(self, player_symbol, epsilon=0.5, gamma=0.5, alpha=0.4, name='Q_player', network=None, train_batch_size=8) -> None:
        super().__init__(player_symbol, epsilon, gamma, alpha, name, network, train_batch_size)
    
    def show_target_values(self, state, alpha, opponent=GreedyPlayer(player_symbol=-1)):
        # return super().show_target_values(state, alpha, opponent)
        return [-999.0]
    
    def generate_network(self):
        # return super().generate_network()
        input = tf.keras.layers.Input(shape=(self.num_rows,self.num_cols,1))
        
        board_flat = InputRemapper()(input)
        board_flat = tf.keras.layers.Flatten()(board_flat)

        board_path = tf.keras.layers.Dense(128,activation='relu')(board_flat)
        board_path = tf.keras.layers.Dropout(0.5)(board_path)
        board_path = tf.keras.layers.Dense(64,activation='relu')(tf.keras.layers.concatenate([board_path,board_flat]))
        board_path = tf.keras.layers.Dropout(0.3)(board_path)
        board_path = tf.keras.layers.Dense(32,activation='relu')(tf.keras.layers.concatenate([board_path,board_flat]))
        ### OLD STUFF ###
        # board_path = tf.keras.layers.Conv2D(filters=32,kernel_size=(4,4),activation='relu',padding='same')(input)
        # board_path = tf.keras.layers.Flatten()(board_path)
        # board_path = tf.keras.layers.Dense(64,activation='relu')(board_path)

        # joined = tf.keras.layers.Dropout(0.5)(board_path)
        # joined = tf.keras.layers.Dense(32,activation='relu')(joined)

        x = tf.keras.layers.Dense(self.num_cols,name=f'{self.num_cols}D_output',activation='tanh')(board_path)
        ### OLD STUFF ###

        model = tf.keras.Model(inputs=input,outputs=x)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['cosine_similarity'],
            loss = tf.keras.losses.MeanSquaredError()
        )
        return model
    def train_network(self,epochs=5):
            states = np.array(self.board_history).reshape(-1,self.num_rows,self.num_cols,1)
            actions = np.row_stack([self.convert_action_to_1_hot(a) for a in self.action_history])
            # a_primes = []
            s_primes = []
            # for s in self.s_prime_history:
                # for i in range(self.num_cols):
                    # state_primes.append(np.array(get_next_board_state(s,a,self.player_symbol)))
                # a_primes.append(self.convert_action_to_1_hot(i))
                # s_primes.append(s.copy())
            

            
            # a_primes = np.array(a_primes)
            state_primes = np.array(self.s_prime_history).reshape(-1,self.num_rows,self.num_cols,1)

            batches = [(i,min(i+self.max_eval_batch_size,len(state_primes))) for i in range(0,len(state_primes),self.max_eval_batch_size)]

            q_star_values = np.concatenate([self.q_prime_network(state_primes[lower:upper]).numpy().reshape(-1,self.num_cols) for lower,upper in batches],axis=0)
            # q_star_values = self.q_prime_network(state_primes,batch_size=self.training_batch_size).numpy().reshape(-1,self.num_cols)
            

            max_q_values = []
            for possible_q_vals,sp in zip(q_star_values,state_primes):
                max_q_values.append(max([v if i in get_possible_moves(sp.reshape(self.num_rows,self.num_cols)) else -999.0 for i,v in enumerate(possible_q_vals)]))

            max_q_values = [i if i != -999.0 else 0 for i in max_q_values]
            target = [(self.alpha)*(q * self.gamma) + (1 - self.alpha) * r if (r != 1 and r != -1) else r for q,r in zip(max_q_values,self.reward_history)]

            # states,actions,targets = self.augment_train_data(states,actions,target)
            targets = target

            batches = [(i,min(i+self.max_eval_batch_size,len(states))) for i in range(0,len(states),self.max_eval_batch_size)]
            raw_targets = np.concatenate([self.q_network(states[lower:upper]).numpy() for lower,upper in batches],axis=0)

            # raw_targets = self.q_network(states,batch_size=self.training_batch_size).numpy()

            for new_val,raw_vals,action in zip(targets,raw_targets,self.action_history):
                raw_vals[action] = new_val
            
            targets = raw_targets
            #TODO Force it to learn invalid states as 0
            print(f""" Before Random Replay:
                States Shape: {states.shape}
                Actions Shape: {actions.shape}
                Len targets: {targets.shape}
                """)
            # print(f""" Before Random Replay:
            #     States Shape: {states.shape}
            #     Actions Shape: {actions.shape}
            #     Len targets: {len(targets)}
            #     """)
            # targets = np.array([max(i,-1.0) if i < 0 else min(i,1.0) for i in targets]).reshape(-1,1) # Clamp target values between -1 and 1
            # targets = np.array(targets).reshape(-1,1)
            print(f'Max Target: {max(targets.flatten())}')
            print(f'Min Target: {min(targets.flatten())}')
            print(f'Std Targets: {np.std(targets)}')

            train_states, test_states, train_targets, test_targets = train_test_split(states,targets)
            train_hist = self.q_network.fit(x=train_states,y=train_targets,epochs=epochs,batch_size=self.training_batch_size,validation_data=(test_states,test_targets))
            new_dct:dict= train_hist.history
            if self.train_history is None:
                self.train_history = new_dct
            else:
                for k,v in new_dct.items():
                    for new_value in v:
                        self.train_history[k].append(new_value)

    def choose_action(self,valid_actions=None,board:np.ndarray=None):  
        if self.q_network is None or self.q_prime_network is None:
            self.q_network = self.generate_network()
            self.q_prime_network = self.generate_network()
            self.copy_model_over() #NOTE: Should I do this? or leave them randomly initialized seperately from eachother
        
        if np.random.random() < self.epsilon:
            self.board_history.append(board.copy())
            action = np.random.choice(valid_actions)
            self.action_history.append(action)
            self.action_reasoning.append(ACTION_REASONING.RANDOM)
            return action
        
        
        # oh_actions = np.row_stack([self.convert_action_to_1_hot(a) for a in range(self.num_cols)])
        # boards = np.row_stack([board.reshape(1,self.num_rows,self.num_cols,1) for i in range(self.num_cols)])
        # # print(f'OH: {oh_actions.shape}\nbrds:{boards.shape}')
        # q_values = self.q_network([boards,oh_actions]).numpy().flatten()
        q_values = self.q_network(board.reshape(-1,self.num_rows,self.num_cols,1)).numpy().flatten()

        q_values = [q if i in valid_actions else -999.0 for i,q in enumerate(q_values)] #Force it to take a valid action
        action = np.random.choice(np.where(q_values == max(q_values))[0])

        self.action_history.append(action)
        self.board_history.append(board.copy())
        self.action_reasoning.append(ACTION_REASONING.POLICY)

        return action
    def show_q_function(self, state):
        # return super().show_q_function(state)
        return self.q_network(state.reshape(-1,self.num_rows,self.num_cols,1)).numpy().flatten()
    

class TransferLearnPlayer(SinglePredictionQPlayer):
    def __init__(self, player_symbol, epsilon=0.5, gamma=0.5, alpha=0.4, name='Q_player', network=None,model_path=None,mdl_stop_idx = 9) -> None:
        super().__init__(player_symbol, epsilon, gamma, alpha, name, network)
        if model_path is None:
            raise ValueError("Please pass a model_path to load transfer learning from")
        self.transfer_model = tf.keras.models.load_model(model_path)
        self.mdl_stop_idx = mdl_stop_idx
    def generate_network(self):
        # return super().generate_network()
        EMBED_DIM = 8
        for l in self.transfer_model.layers:
            l.trainable = False
        # self.transfer_model.get_layer('4x4_conv').trainable = False

        action_input = tf.keras.layers.Input(shape=(1))
        action_path = tf.keras.layers.Embedding(input_dim=7,output_dim=EMBED_DIM,name='action_encoder')(action_input)

        transfer_path = tf.keras.layers.Dense(32,activation='tanh')(self.transfer_model.layers[self.mdl_stop_idx].output)
        transfer_path = tf.keras.layers.Dense(EMBED_DIM)(transfer_path)
        joined_path = tf.keras.layers.Multiply()([action_path,transfer_path])

        joined_path = tf.keras.layers.Dense(64)(joined_path)
        joined_path = tf.keras.layers.Dense(16)(joined_path)

        joined_path = tf.keras.layers.Dense(1,name='1d_output',activation='tanh')(joined_path)
        # joined_path = tf.keras.layers.Activation('linear',name='tanh_activation')(joined_path)

        mdl_2 = tf.keras.Model(inputs=[self.transfer_model.input,action_input],outputs=[joined_path])
        mdl_2.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=[tf.keras.metrics.CosineSimilarity()]
        )
        # mdl_2.output.
        # print(f'{self.name} Model:\n\t{mdl_2.summary()}')
        return mdl_2

class HumanPlayer(DeepQPlayer):
    def __init__(self, player_symbol, epsilon=0.5, gamma=0.5, alpha=0.4, name='Q_player', network=None, training_batch_size=256, max_eval_batch_size=4096) -> None:
        super().__init__(player_symbol, epsilon, gamma, alpha, name, network, training_batch_size, max_eval_batch_size)
    
    def choose_action(self, valid_actions=None, board: np.ndarray = None, save_choices=False):
        # return super().choose_action(valid_actions, board, save_choices)
        action = input('Select a column index')
        keep_going = not action.isnumeric()
        if not keep_going:
            if int(action) > self.num_cols - 1:
                keep_going = True
        while not action.isnumeric():
            action = input ('Must be a number')
            keep_going = not action.isnumeric()
            if not keep_going:
                if int(action) > self.num_cols - 1:
                    keep_going = True
        return int(action)
    def show_q_function(self, state):
        # return super().show_q_function(state)
        return [-999] * self.num_cols
            

class Board:
    def __init__(self,p1:BasePlayer,p2:BasePlayer,num_rows=6,num_cols=7,cooling_func = lambda x,y :x):
        self.num_rows= num_rows
        self.num_cols = num_cols
        self.p1 = p1
        self.p2 = p2
        self.p1.num_rows = num_rows
        self.p1.num_cols = num_cols
        self.p2.num_rows = num_rows
        self.p2.num_cols = num_cols
        self.cooling_func = cooling_func
        self.board = np.zeros(shape=(num_rows,num_cols),dtype=int)

    def check_win(self):
        
        horizontal_kernel = np.array([[ 1, 1, 1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(4, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
        for kernel in detection_kernels:

            a = convolve2d(self.board,kernel,mode='valid')
            if( (a == 4).any()):
                return 1
            if ((a == -4).any()):
                return -1
            
        if(len(self.get_valid_moves()) == 0): return None
        return 0
    def get_valid_moves(self):
        return [i for i,j in enumerate(self.board[0]) if j == 0]
    def reset(self):
        self.board = np.zeros(shape=(self.num_rows,self.num_cols))
    def __str__(self) -> str:
        visualize(self.board)
        return ""
    def place(self,move,player_symbol=1):
        if move not in self.get_valid_moves():
            raise RuntimeError(f"Invalid Action {move} with board state: \n{str(self.board)}")
        col = self.board[:,move]
        idx = int(np.where(col == 0)[0][-1])
        col[idx] = player_symbol

    def train_agents(self,n_iterations=500,verbose=0,copy_iterations=15000,train_iterations = 15000,training_epochs=10,alpha_cooling_func=lambda x,y:x,reward_func=lambda b,p:0.0):
        for i in (pbar:= tqdm(range(n_iterations + 1))):
            # if i % 1000 == 0:
            pbar.set_description(f'P1 Epsilon: {round(self.cooling_func(self.p1.initial_epsilon,i),ndigits=2)}, P2 Epsilon:{round(self.cooling_func(self.p2.initial_epsilon,i),ndigits=2)},P1 Alpha: {self.p1.alpha}')
            s1 = None
            s1_prime = None
            s2 = None
            s2_prime = None
            self.p1.epsilon = self.cooling_func(self.p1.initial_epsilon,i)
            self.p2.epsilon = self.cooling_func(self.p2.initial_epsilon,i)
            if isinstance(self.p1,DeepQPlayer):

                self.p1.alpha = alpha_cooling_func(self.p1.alpha,i)
            if isinstance(self.p2,DeepQPlayer):
                self.p2.alpha = alpha_cooling_func(self.p2.alpha,i)

            while self.check_win() == 0:
                valid_actions = self.get_valid_moves()
                
                p1_action = self.p1.choose_action(valid_actions=valid_actions,board=self.board)
                s1 = self.board.copy()
                self.place(p1_action,player_symbol=self.p1.player_symbol)
                p1_ev = reward_func(self.board,self.p1.player_symbol)
                if verbose >= 1:
                    visualize(self.board)
                if self.check_win() == self.p1.player_symbol:
                    s1_prime = self.board.copy()
                    self.p1.give_reward(1.0,s1,s1_prime,p1_action)
                    self.p2.give_reward(-1.0,s2,s1_prime,p2_action)
                    break
                elif self.check_win() is None:
                    s1_prime = self.board.copy()
                    self.p1.give_reward(0.0,s1,s1_prime,p1_action)
                    self.p2.give_reward(0.0,s2,s1_prime,p2_action)
                    break

                valid_actions = self.get_valid_moves()
                p2_action = self.p2.choose_action(valid_actions=valid_actions,board=self.board)
                s2_prime = self.board.copy()

                self.place(p2_action,player_symbol=self.p2.player_symbol)
                p2_ev = reward_func(self.board,self.p2.player_symbol) # Get p2 EV after p2 has played
                s1_prime = self.board.copy()

                if verbose >= 1:
                    visualize(self.board)
                
                
                if self.check_win() == self.p2.player_symbol:
                    s2 = s2_prime
                    s2_prime = self.board.copy()
                    self.p1.give_reward(-1.0,s1,s2_prime,p1_action)
                    self.p2.give_reward(1.0,s2,s2_prime,p2_action)
                    break
                elif self.check_win() is None:
                    s2 = s2_prime
                    s2_prime = self.board.copy()
                    self.p1.give_reward(0.0,s1,s2_prime,p1_action)
                    self.p2.give_reward(0.0,s2,s2_prime,p2_action)
                    break
                
                
                self.p1.give_reward(p1_ev,s1,s1_prime,p1_action)
                if s2 is not None:
                    self.p2.give_reward(p2_ev,s2,s2_prime,p2_action)
                s2 = s2_prime.copy()
            
            if i % train_iterations == 0 and i != 0:
                if isinstance(self.p1,DeepQPlayer):
                    self.p1.train_network(epochs=training_epochs)
                    self.p1.reset()
                if isinstance(self.p2,DeepQPlayer):
                    self.p2.train_network(epochs=training_epochs)
                    self.p2.reset()
            if i % copy_iterations == 0 and i != 0:
                if isinstance(self.p1,DeepQPlayer):
                    self.p1.copy_model_over()
                if isinstance(self.p2,DeepQPlayer):
                    self.p2.copy_model_over()
                    # self.p2.reset()
            if not isinstance(self.p1,DeepQPlayer):
                self.p1.reset()
            if not isinstance(self.p2,DeepQPlayer):
                self.p2.reset()
                
            self.reset()
           

    def play_agents(self,verbose=1,return_boards = False):
        past_board_states = []
        while self.check_win() == 0 and self.check_win() is not None:
            valid_actions = self.get_valid_moves()
            if verbose >= 2 and isinstance(self.p1,QPlayer):
                print(f'QFunction: {list(np.round(self.p1.show_q_function(self.board),2))}')
                if isinstance(self.p1,DeepQPlayer):
                    print(f'Targets (alpha=0): {np.round(self.p1.show_target_values(self.board,alpha=0),2)}')
                    print(f'Targets (alpha=1): {np.round(self.p1.show_target_values(self.board,alpha=1),2)}')
            p1_action = self.p1.choose_action(valid_actions=valid_actions,board=self.board)
            self.place(p1_action,player_symbol=self.p1.player_symbol)
            past_board_states.append(self.board.copy())
            if verbose:
                print(self)
            if self.check_win() == self.p1.player_symbol:
                if verbose:
                    print(f"Player {self.p1.player_symbol} Wins!")
                return self.p1.player_symbol
            elif self.check_win() is None:
                if verbose:
                    print("Tie!")
                    return None
            else:
                valid_actions = self.get_valid_moves()
                if verbose >= 2 and isinstance(self.p2,QPlayer):
                    print(self.p2.show_q_function(self.board))
                p2_action = self.p2.choose_action(valid_actions=valid_actions,board=self.board)
                self.place(p2_action,player_symbol=self.p2.player_symbol)
                past_board_states.append(self.board.copy())

                if verbose:
                    print(self)

                if self.check_win() == self.p2.player_symbol:
                    if verbose:
                        print(f"Player {self.p2.player_symbol} Wins!")
                    return self.p2.player_symbol

        if return_boards:
            return past_board_states
        else:
            return self.check_win()

def generate_random_board(num_rows=6,num_cols=7,num_moves=None):
    r1 = RandomPlayer(player_symbol=1)
    r2 = RandomPlayer(player_symbol=-1)
    b = Board(r1,r2,num_rows=num_rows,num_cols=num_cols)
    if num_moves is None:
        b.play_agents(verbose=0)
        return b.board
    else:
        for i in range(num_moves):
            b.place(r1.choose_action(b.get_valid_moves()),r1.player_symbol)
            b.place(r2.choose_action(b.get_valid_moves()),r2.player_symbol)
        return b.board
    



def alpha_cooling_func(x,y):
    if y < 10000:
        return x
    elif y < 25000:
        return 0.2
    elif y < 50000:
        return 0.4
    elif y < 75000:
        return 0.6
    else:
        return 1.0
if __name__ == "__main__":
    def custom_cooling(eps,itr):
        if itr < 25000:
            return eps
        elif itr < 50000:
            return 0.6
        elif itr < 75000:
            return 0.4
        elif itr < 100000:
            return 0.2
        else:
            return 0.05

    q_multi_d2rl = MultiPredictionQPlayer(player_symbol=1,name='Q_Multi_D2RL',epsilon=0.5,alpha=1.0,gamma=0.95,train_batch_size=128)
    rand_multi_d2rl = MultiPredictionQPlayer(player_symbol=1,name='Rand_Multi_D2RL',epsilon=0.5,alpha=1.0,gamma=0.95,train_batch_size=128)

    q_single_d2rl = SinglePredictionQPlayer(player_symbol=1,name='Q_Single_D2RL',epsilon=1,alpha=1.0,gamma=0.95,train_batch_size=128)
    rand_single_d2rl = SinglePredictionQPlayer(player_symbol=1,name='Rand_Single_Encoding',epsilon=0.5,alpha=1.0,gamma=0.95,train_batch_size=128)


    q2_multi = MultiPredictionQPlayer(player_symbol=-1,name='Q2_VS_Q_V3_D2RL',epsilon=0.5,alpha=1.0,gamma=0.95,train_batch_size=128)
    q2_single = SinglePredictionQPlayer(player_symbol=-1,name='Q2_VS_Q_V3_D2RL',epsilon=1,alpha=1.0,gamma=0.95,train_batch_size=128)
    # q2 = SinglePredictionQPlayer(player_symbol=-1,name='q2_input_mapper_vsq_small_train',epsilon=0.5,alpha=0.0,gamma=0.7,train_batch_size=128)
    r2 = RandomPlayer(player_symbol = -1)
    monte_2 = MonteCarloPlayer(player_symbol=-1)

    g2 = GreedyPlayer(player_symbol=-1)

    # b = Board(q1,q2,num_rows=6,num_cols=7)

    def write_train_board(b:Board):
        try:
            print(f'Training {type(b.p1)} Vs. {type(b.p2)}')
            # b.train_agents(n_iterations=750000,verbose=0,train_iterations=1000,copy_iterations=2000,training_epochs=1,alpha_cooling_func=alpha_cooling_func,reward_func=evaluate_board) Pre 4/22
            b.train_agents(n_iterations=200000,verbose=0,train_iterations=1000,copy_iterations=2000,training_epochs=1,reward_func=evaluate_board)

        except KeyboardInterrupt as e:
            print('Training Stoppped Early...')
        finally:
            b.p1.savePolicy()

    b1 = Board(rand_single_d2rl,q2_single,cooling_func=custom_cooling)
    # b2 = Board(rand_multi_d2rl,r2)
    # b3 = Board(q_single_d2rl,q2_single)
    # b4 = Board(rand_single_d2rl,r2)

    write_train_board(b1)
    # write_train_board(b2)
    # write_train_board(b3)
    # write_train_board(b4)
    
        

