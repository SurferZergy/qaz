import logging
import time
import coloredlogs

from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.qtensorflow2.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters':10,
    'numEps': 100, #100              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.49,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000, #64000 = 1000 bataches for training #200000   # Number of game examples to train the neural networks.
    'numMCTSSims': 25, #25          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40, #40         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './ttnt/',
    'load_model': False,
    'load_folder_file': ('./ttn/t','best'),
    'numItersForTrainExamplesHistory': 20, #20

})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(4)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')

    start = time.time()
    c.learn()
    stop = time.time()
    print(f"Training time: {stop - start}s")



if __name__ == "__main__":
    main()
