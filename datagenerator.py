from deepq import RandomPlayer,Board,get_next_board_state
from multiprocessing import pool
import numpy as np
from tqdm.auto import tqdm
import os
import json
from concurrent.futures import ThreadPoolExecutor

ROOT_DATA_FOLDER = 'genuse_data/'

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def generate_random_board(num_rows=6,num_cols=7,num_moves=None):
    r1 = RandomPlayer(player_symbol=1)
    r2 = RandomPlayer(player_symbol=-1)
    b = Board(r1,r2,num_rows=num_rows,num_cols=num_cols)
    if num_moves is None:
        b.play_agents(verbose=0)
        return b.board
    else:
        num_tries = 0
        for i in range(num_moves):
            b.place(r1.choose_action(b.get_valid_moves()),r1.player_symbol)
            b.place(r2.choose_action(b.get_valid_moves()),r2.player_symbol)
        while(b.check_win() != 0):
            b.reset()
            for i in range(num_moves):
                b.place(r1.choose_action(b.get_valid_moves()),r1.player_symbol)
                b.place(r2.choose_action(b.get_valid_moves()),r2.player_symbol)
                num_tries += 1
        # print(f'Num tries: {num_tries}')
        return b.board
    
def play(board:np.ndarray,p1,p2):
    b = Board(p1=p1,p2=p2,num_rows=len(board[0]),num_cols=len(board))
    b.reset()
    b.board = board
    return b.play_agents(verbose=0)

def get_targets_for_board(board,num_iterations=500):
    player_vals,player_counts = np.unique(board,return_counts=True)
    tmp_dct = {k:v for k,v in zip(player_vals,player_counts)}
    red_pieces = tmp_dct.get(1,0)
    yellow_pieces = tmp_dct.get(-1,0)

    board_dict = {}
    if red_pieces == yellow_pieces:
        p1 = RandomPlayer(player_symbol=-1)
        p2 = RandomPlayer(player_symbol=1)
        for action in range(len(board[0])):
            board_dict[action] = get_next_board_state(board,action,1)
        
    else:
        p1 = RandomPlayer(player_symbol=1)
        p2 = RandomPlayer(player_symbol=-1)
        for action in range(len(board[0])):
            board_dict[action] = get_next_board_state(board,action,-1)

    for action,new_board in board_dict.items():
        if new_board is None:
            board_dict[action] = {-1:0,0:0,1:0}
            continue 
        winners = [play(new_board.copy(),p1,p2) for i in range(num_iterations)]
        vals,counts = np.unique([v if v is not None else 0 for v in winners],return_counts=True)
        v = {k:float(v/num_iterations) for k,v in zip(vals,counts)}
        board_dict[action] = {-1:v.get(-1,0.0),0:v.get(0,0.0),1:v.get(1,0.0)}
    return board_dict


def write_random_board(num_pieces,num_rows,num_cols):
    # print('Im a Thread Starting')
    brd = generate_random_board(num_moves=num_pieces,num_rows=num_rows,num_cols=num_cols)
    brd_str = str(brd.flatten()) + '.json'
    brd_str = os.path.join(ROOT_DATA_FOLDER,brd_str)
    num_attempts = 0
    while os.path.exists(brd_str):
        brd = generate_random_board(num_moves=num_pieces,num_rows=num_rows,num_cols=num_cols)
        brd_str = str(brd.flatten()) + '.json'
        brd_str = os.path.join(ROOT_DATA_FOLDER,brd_str)
        num_attempts += 1
        if num_attempts > 100:
            return
    with open(brd_str,'w') as fp:
        targets = get_targets_for_board(brd,num_iterations=500)
        to_json = {'targets':targets,'board':brd}
        json.dump(to_json,fp,cls=NpEncoder)
    # print('Im a thread finishing')
    
class DataGenerator:
    def __init__(self,piece_numbers,num_boards_per_pieces=5000,generate_on_construct = False,test_size=0.2,num_rows=6,num_cols=7):
        self.pieces = piece_numbers
        self.boards_per = num_boards_per_pieces
        
        self.train_labels = None
        self.train_data = None
        self.test_data = None
        self.test_labels = None
        self.test_size = test_size

        self.num_rows = num_rows
        self.num_cols = num_cols

        if generate_on_construct:
            pass
    def _generate_data_dict(self,num_data):
        data = []
        target = []

        for num_pieces in tqdm(self.pieces,desc='Pieces'):
            for i in tqdm(range(num_data),desc='Boards'):
                brd = generate_random_board(num_moves=num_pieces,num_rows=self.num_rows,num_cols=self.num_cols)
                data.append(brd.copy())
                target.append(get_targets_for_board(brd))
        
        data = np.array(data)
        target = np.array(target)
        p = np.random.permutation(len(data)) #Randomly shuffle the data

        return data[p],target[p]
    
    def generate_data(self):
        """Returns train_data,test_data,train_targets,test_targets from data generator object"""
        test_num = int(self.test_size * self.boards_per)
        train_num = self.boards_per - test_num
        print(f"Generating boards for {self.pieces} possible pieces with {test_num} test set size and {train_num} train set size")
        train_data,train_targets = self._generate_data_dict(train_num)
        test_data,test_targets = self._generate_data_dict(test_num)

        return train_data,test_data,train_targets,test_targets

    def write_board_batch(self):
        
        pieces = []
        rows = []
        cols = []
        data = []
        for num_pieces in self.pieces:
            for i in range(self.boards_per):
                pieces.append(num_pieces)
                rows.append(self.num_rows)
                cols.append(self.num_cols)
                data.append((num_pieces,self.num_rows,self.num_cols))
        # print(len(data))
        # with ThreadPoolExecutor(max_workers=None) as executor:
        #     list(executor.map(write_random_board, pieces,rows,cols))
        with pool.Pool(processes=24) as p:
            results = p.starmap(write_random_board,tqdm(data,total=len(data),desc=f'Writing to {ROOT_DATA_FOLDER}'))
    def write_random_boards(self,nboards):
        data = []
        for i in range(nboards):
            data.append((np.random.randint(2,19),self.num_rows,self.num_cols))
        
        with pool.Pool(processes=24) as p:
            results = p.starmap(write_random_board,data)
 

        
if __name__ == "__main__":
    # print(write_random_board(5,6,7))
    data_gen = DataGenerator(piece_numbers=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],num_boards_per_pieces=4000)
    data_gen.write_random_boards(150)