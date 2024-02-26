import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from enum import Enum
import pickle

import gc

gc.collect()

def visualize(board):
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


AGENT_VERBOSE = True
class BasePlayer:
    def __init__(self,player_symbol) -> None:
        self.player_symbol = player_symbol
        self.num_rows = 0
        self.num_cols = 0
        self.epsilon = 0.0
        self.initial_epsilon = 0.0
    def choose_action(self,valid_actions=None,board=None):
        raise NotImplementedError("Player class is virtual, please inherent from it")
    def give_reward(self,reward,s,s_prime,action):
        raise NotImplementedError("Player class is virtual, please inherent from it")
    def reset(self):
        raise NotImplementedError("Player class is virtual, please inherent from it")
    
class RandomPlayer(BasePlayer):
    def __init__(self, player_symbol) -> None:
        super().__init__(player_symbol)
    def choose_action(self, valid_actions=None, board=None):
        return np.random.choice(valid_actions)
    def give_reward(self, reward,s,s_prime,action):
        pass
    def reset(self):
        pass

def get_next_board_state(board,action,player_symbol):
    try:
        _board = board.copy()
        col = _board[:,action]
        idx = int(np.where(col == 0)[0][-1])
        col[idx] = player_symbol
        return _board
    except:
        return None
def get_possible_moves(board):
    return [i for i,j in enumerate(board[0]) if j == 0]

class ACTION_REASONING(Enum):
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
        return self.q_table.get(self.get_board_hash(state),"Never Seen State")
    def get_board_hash(self,board:np.ndarray):
        # return str(board.flatten())
        return board.tobytes()
    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.q_table, fw)
        fw.close()

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
        



class DeepQPLayer(QPlayer):
    def __init__(self, player_symbol, epsilon=0.5, gamma=0.5, alpha=0.4, name='Q_player',network=None) -> None:
        super().__init__(player_symbol, epsilon, gamma, alpha, name)
        # self.q_network = network.
        if network is not None:
            self.q_network = tf.keras.models.clone_model(network)
            self.q_prime_network = tf.keras.models.clone_model(network)
        else:
            self.q_network = None
            self.q_prime_network=None
        self.reward_history = []
        self.s_prime_history = []

        
    def copy_model_over(self):
        self.q_prime_network.set_weights(self.q_network.get_weights())


    def show_q_function(self, state):
        # return super().show_q_function(state)
        return np.array([self.q_network([state.reshape(1,self.num_rows,self.num_cols,1),np.array(self.convert_action_to_1_hot(a)).reshape(-1,self.num_cols)]) for a in range(self.num_cols)]).flatten()


    def generate_network(self):

        board_input = tf.keras.layers.Input(shape=(self.num_rows,self.num_cols,1))
        action_input = tf.keras.layers.Input(shape=(self.num_cols))

        board_path = tf.keras.layers.Conv2D(filters=12,kernel_size=(3,3))(board_input)

        board_path = tf.keras.layers.Activation('relu')(board_path)
        board_path = tf.keras.layers.Flatten()(board_path)
        board_path = tf.keras.layers.concatenate([action_input,board_path])
        board_path = tf.keras.layers.Dense(128)(board_path)
        board_path = tf.keras.layers.Dense(1)(board_path)
        board_path = tf.keras.layers.Activation('tanh')(board_path)

        model = tf.keras.Model(inputs=[board_input,action_input],outputs=board_path)
        model.compile(loss=tf.losses.mean_squared_error)
        return model

    def convert_action_to_1_hot(self,action):
        _arr = np.array([0] * self.num_cols)
        _arr [action] = 1
        return _arr
    
    
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
        
        
        oh_actions = np.row_stack([self.convert_action_to_1_hot(a) for a in range(self.num_cols)])
        boards = np.row_stack([board.reshape(1,self.num_rows,self.num_cols,1) for i in range(self.num_cols)])
        # print(f'OH: {oh_actions.shape}\nbrds:{boards.shape}')
        q_values = self.q_network([boards,oh_actions]).numpy().flatten()

        q_values = [q if i in valid_actions else -999.0 for i,q in enumerate(q_values)] #Force it to take a valid action
        action = np.random.choice(np.where(q_values == max(q_values))[0])

        self.action_history.append(action)
        self.board_history.append(board.copy())
        self.action_reasoning.append(ACTION_REASONING.POLICY)

        return action
        
    def give_reward(self,reward,s=None,s_prime=None,action=None,n_train_epochs=3):
        # raise NotImplementedError("Player class is virtual, please inherent from it")
        self.s_prime_history.append(s_prime)
        self.reward_history.append(reward)
    
    def train_network(self,epochs=10):
        states = np.array(self.board_history).reshape(-1,self.num_rows,self.num_cols,1)
        actions = np.row_stack([self.convert_action_to_1_hot(a) for a in self.action_history])
        # print(actions.shape)
        # if len(self.s_prime_history) != len(self.board_history):
        #     self.s_prime_history.append(self.board_history[-1])
        

        a_primes = []
        s_primes = []
        for s in self.s_prime_history:
            for i in range(self.num_cols):
                # state_primes.append(np.array(get_next_board_state(s,a,self.player_symbol)))
                a_primes.append(self.convert_action_to_1_hot(i))
                s_primes.append(s.copy())
        

        
        a_primes = np.array(a_primes)
        state_primes = np.array(s_primes).reshape(-1,self.num_rows,self.num_cols,1)
#         print(f"""
# state_prime: {state_primes.shape}
# states: {states.shape}
# a_prime:{a_primes.shape}
#               """)

        q_star_values = self.q_prime_network([state_primes,a_primes]).numpy().reshape(-1,self.num_cols)
        

        # q_star_values = np.array([a for q in q_star_values for a in ])
        max_q_values = []
        for possible_q_vals,sp in zip(q_star_values,state_primes):
            max_q_values.append(max([v if i in get_possible_moves(sp.reshape(self.num_rows,self.num_cols)) else -999.0 for i,v in enumerate(possible_q_vals)]))

        max_q_values = [i if i != -999.0 else 0 for i in max_q_values]
        target = [q * self.gamma + r for q,r in zip(max_q_values,self.reward_history)]

        #TODO Force it to learn invalid states as 0 
        target = np.array([max(i,-1.0) if i < 0 else min(i,1.0) for i in target]).reshape(-1,1) # Clamp target values between -1 and 1

        print(f'Max Target: {max(target.flatten())}')
        print(f'Min Target: {min(target.flatten())}')

        self.q_network.fit(x=[states,actions],y=target,epochs=epochs)

    def savePolicy(self):
        self.q_network.save(f'q_network_{self.name}')
        self.q_prime_network.save(f'q_prime_network_{self.name}')
    def loadPolicy(self, file):
        # return super().loadPolicy(file)
        self.q_network = tf.keras.models.load_model(file)
        self.q_prime_network = tf.keras.models.load_model(file)
        print(f'{self.name} Loaded Q and Q Prime networks from file {file}')
    def reset(self):
        # raise NotImplementedError("Player class is virtual, please inherent from it")
        self.action_reasoning.clear()
        self.board_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        self.s_prime_history.clear()
        
        
        #NOTE: Might want to remove this to have model retrain on states that it has already seen before? Up to some max value (maybe same value that we copy model over with)

class Board:
    def __init__(self,p1:BasePlayer,p2:BasePlayer,num_rows=7,num_cols=7,cooling_func = lambda x,y :x):
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
        
        if(len(self.get_valid_moves()) == 0): return None
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
            
            # print(a.any())
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

    def train_agents(self,n_iterations=500,verbose=0,copy_iterations=15000,train_iterations = 15000,training_epochs=10):
        for i in (pbar:= tqdm(range(n_iterations))):
            # if i % 1000 == 0:
            pbar.set_description(f'P1 Epsilon: {round(self.cooling_func(self.p1.initial_epsilon,i),ndigits=2)}, P2 Epsilon:{round(self.cooling_func(self.p2.initial_epsilon,i),ndigits=2)}')
            s1 = None
            s1_prime = None
            s2 = None
            s2_prime = None
            self.p1.epsilon = self.cooling_func(self.p1.initial_epsilon,i)
            self.p2.epsilon = self.cooling_func(self.p2.initial_epsilon,i)
            while self.check_win() == 0:
                valid_actions = self.get_valid_moves()
                
                p1_action = self.p1.choose_action(valid_actions=valid_actions,board=self.board)
                s1 = self.board.copy()

                self.place(p1_action,player_symbol=self.p1.player_symbol)
                if verbose == 1:
                    visualize(self.board)

                if self.check_win() == self.p1.player_symbol:
                    s1_prime = self.board.copy()
                    self.p1.give_reward(1.0,s1,s1_prime,p1_action)
                    self.p2.give_reward(-1.0,s2,s1_prime,p2_action)
                    break
                elif self.check_win() is None:
                    s1_prime = self.board.copy()
                    self.p1.give_reward(0.5,s1,s1_prime,p1_action)
                    self.p2.give_reward(0.5,s2,s1_prime,p2_action)
                    break

                valid_actions = self.get_valid_moves()
                p2_action = self.p2.choose_action(valid_actions=valid_actions,board=self.board)
                s2_prime = self.board.copy()

                self.place(p2_action,player_symbol=self.p2.player_symbol)
                s1_prime = self.board.copy()

                if verbose == 1:
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
                    self.p1.give_reward(0.5,s1,s2_prime,p1_action)
                    self.p2.give_reward(0.5,s2,s2_prime,p2_action)
                    break
                
                
                self.p1.give_reward(0.0,s1,s1_prime,p1_action)
                if s2 is not None:
                    self.p2.give_reward(0.0,s2,s2_prime,p2_action)
                s2 = s2_prime

            winner = self.check_win()
            # if isinstance(self.p1,DeepQPLayer):
            #     self.p1.give_reward(winner * self.p1.player_symbol if winner is not None else 0.5) #Cheat to give reward
            # if isinstance(self.p2,DeepQPLayer):
            #     self.p2.give_reward(winner * self.p2.player_symbol if winner is not None else 0.5)
            

            if i % copy_iterations == 0 and i != 0:
                if isinstance(self.p1,DeepQPLayer):
                    self.p1.copy_model_over()
                if isinstance(self.p2,DeepQPLayer):
                    self.p2.copy_model_over()
                    # self.p2.reset()
            if not isinstance(self.p1,DeepQPLayer):
                self.p1.reset()
            if not isinstance(self.p2,DeepQPLayer):
                self.p2.reset()
            if i % train_iterations == 0 and i != 0:
                if isinstance(self.p1,DeepQPLayer):
                    self.p1.train_network(epochs=training_epochs)
                    self.p1.reset()
                if isinstance(self.p2,DeepQPLayer):
                    self.p2.train_network(epochs=training_epochs)
                    self.p2.reset()

                gc.collect()
                
            self.reset()
           

    def play_agents(self,verbose=True,return_boards = False):
        past_board_states = []
        while self.check_win() == 0 and self.check_win() is not None:
            valid_actions = self.get_valid_moves()
            if verbose == 2 and isinstance(self.p1,QPlayer):
                print(self.p1.show_q_function(self.board))
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
                if verbose == 2 and isinstance(self.p2,QPlayer):
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



if __name__ == "__main__":
    def custom_cooling(eps,itr):
        if itr < 250000 :
            return eps
        elif itr < 500000 :
            return eps / 2.0
        elif itr < 750000 :
            return eps / 4.0
        else:
            return eps / 8.0
    
    q1 = DeepQPLayer(player_symbol=1,name='q_1_cooling_deep_v2',epsilon=0.5)
    # q1 = QPlayer(player_symbol=-1,name='q_1_cooling')
    q2 = QPlayer(player_symbol=-1,name='q_2_cooling')
    r2 = RandomPlayer(player_symbol = -1)

    q1.loadPolicy('q_network_q_1_cooling_deep')

    b = Board(q1,r2,num_rows=4,num_cols=4,cooling_func=custom_cooling)

    b.reset()
    q1.reset()
    q2.reset()
    b.train_agents(n_iterations=1000000,verbose=0,train_iterations=2500,copy_iterations=10000)
    q1.savePolicy()
    q2.savePolicy()
