import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.qtensorflow2.NNet import NNetWrapper as QNNet
from othello.tensorflow2.NNet import NNetWrapper as NNet
from othello.tfq.NNet3 import NNetWrapper as PQCNNet
from othello.tfq.NNet import NNetWrapper as PQCNNet8

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = True  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = False

if mini_othello:
    g = OthelloGame(4)
else:
    g = OthelloGame(8)

# all players
rp = RandomPlayer(g).play
gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play



# nnet players
n1 = NNet(g)
if mini_othello:
    # n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
    n1.load_checkpoint('./clas44/', 'best')
else:
    # n1.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')
    n1.load_checkpoint('./25iter/','best')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:


    # this is for class
    # n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
    # args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    # mcts2 = MCTS(g, n2, args2)
    # n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    # This is for TTN
    # n2 = QNNet(g)
    # n2.load_checkpoint('./ttn2/', 'best')
    # args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    # mcts2 = MCTS(g, n2, args2)
    # n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    # this one for pqc3
    # n2 = PQCNNet(g)
    # n2.load_checkpoint('./pqc3_2/', 'best')
    # args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    # mcts2 = MCTS(g, n2, args2)
    # n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    # this one for pqc2
    # n2 = PQCNNet8(g)
    # n2.load_checkpoint('./pqc2_3/', 'best')
    # args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    # mcts2 = MCTS(g, n2, args2)
    # n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    # this one for pqcSM
    n2 = PQCNNet8(g)
    n2.load_checkpoint('./pqcSM/', 'best')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(n1p, player2, g, display=OthelloGame.display)

print(arena.playGames(10, verbose=True))
# print(arena.playGames(10, verbose=False))