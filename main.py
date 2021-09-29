import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np
import random
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN
from IPython.display import clear_output
import chess 

# Global variables
WIN_REWARD = 1
GAME_LENGTH = 200
DRAW_PENALTY = 0
INVALID_MOVE = 0.00001
VALID_MOVE = 0.00001

# Chess enviroment
class ChessGameEnv_V2(Env):
    def __init__(self):
        # Set spaces
        self.action_space = MultiDiscrete([64,64])
        self.observation_space = Box(-6,6, shape=(8,8), dtype=int)
        
        # Set board
        self.board = chess.Board()
        self.board.reset()
        
        # Set state
        self.state = self.piece_map_v2()
        
        # Set variables
        self.game_length = GAME_LENGTH
        self.fen = self.board.fen()
        self.done = False
        self.info = {'move': False, 'result': '', 'details': 0}
        
    def step(self, action):
        self.info['move'] = False
        self.info['details'] = ''
        reward = 0
        
        _from = action[0]
        _to = action[1]
        
        # Try to find legal move
        try:
            move = self.board.find_move(from_square = _from, to_square = _to)
        except:
            self.info['details'] = "Invalid move"
            return self.state, -INVALID_MOVE, self.done, self.info
        
        self.info['move'] = True
        self.board.push(move)
        reward += VALID_MOVE
        
        # Check status
        if self.check_if_ended() == "Win":
            self.done = True
            reward += WIN_REWARD
            self.info['result'] = 'Win'
            return self.state, self.reward, self.done, self.info
        elif self.check_if_ended() == "Loss":
            self.done = True
            reward -= WIN_REWARD
            self.info['result'] = 'Loss'
            return self.state, self.reward, self.done, self.info
        elif self.check_if_ended() == "Draw":
            self.done = True
            reward -= DRAW_PENALTY
            self.info['result'] = 'Draw'
            return self.state, self.reward, self.done, self.info
        
        # Switch player
        legal_moves = [move for move in self.board.legal_moves]
        move = random.choice(legal_moves)
        
        self.board.push(move)
        
        # Check status
        if self.check_if_ended() == "Win":
            self.done = True
            reward += WIN_REWARD
            self.info['result'] = 'Win'
            return self.state, self.reward, self.done, self.info
        elif self.check_if_ended() == "Loss":
            self.done = True
            reward -= WIN_REWARD
            self.info['result'] = 'Loss'
            return self.state, self.reward, self.done, self.info
        elif self.check_if_ended() == "Draw":
            self.done = True
            reward -= DRAW_PENALTY
            self.info['result'] = 'Draw'
            return self.state, self.reward, self.done, self.info
    
        # Set states
        self.game_length -= 1
        self.state = self.piece_map_v2()
        self.fen = self.board.fen()
        return self.state, reward, self.done, self.info
    
    def render(self):
        print(self.board)
        print(self.fen)
    def reset(self):
        
        # Set spaces
        self.action_space = MultiDiscrete([64,64])
        self.observation_space = Box(-6,6, shape=(8,8), dtype=int)
        
        # Set board
        self.board = chess.Board()
        self.board.reset()
        
        # Set state
        self.state = self.piece_map_v2()
        
        # Set variables
        self.game_length = GAME_LENGTH
        self.fen = self.board.fen()
        self.reward = 0
        self.done = False
        self.info = {'move': False, 'result': '', 'details': 0}
        
        return self.state
        
    def check_if_ended(self):
        # If game length limit
        if self.game_length == 0:
            return "Draw"
        
        # If draw
        if self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.can_claim_draw():
            return "Draw"
        
        # If win
        if self.board.is_checkmate():
            if self.board.outcome().winner == True:
                return "Win"
            else:
                return "Loss"
        
        return False
                
    def piece_map_v2(self):
        array = np.zeros((8, 8), dtype=int)
        
        for square, piece in self.board.piece_map().items():
            rank, file = chess.square_rank(square), chess.square_file(square)
            piece_type, color = piece.piece_type, piece.color

            offset = 1 if color == chess.WHITE else -1

            array[rank, file] = piece_type * offset
        return array
        
# Set the enviroment
env = ChessGameEnv_V2()

# Set log path for Tensorboard
log_path = os.path.join('Training', 'Logs')

# Set the model
model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0001, tensorboard_log=log_path)#, gae_lambda=0.35, )

# Train the model
model.learn(total_timesteps=5000000)

# Test the enviroment by playing a game 
obs = env.reset()
done = False
score = 0

false_actions = np.zeros(64)

for e in range(1,2):
    done = False
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if info['move'] == True:
            clear_output(wait=True)
            env.render()
            print("Reward: ", reward)
            print("\n")
        else: 
            false_actions[action] += 1
        score += reward
        
print("\nScore: ", score)
