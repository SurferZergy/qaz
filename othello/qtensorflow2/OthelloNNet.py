import sys
sys.path.append('..')
from utils import *

import argparse
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import pennylane as qml
from pennylane import numpy as np

n_qubits = 12
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, theta):
    N = n_qubits  ### to draw our circuit, set break pt here, and type in console:
                    # qml.drawer.use_style("black_white")
                    # f, a = qml.draw_mpl(qnode)(inputs, theta)

    # Encoding of 16 dim array
    for j in range(N):
        qml.RY(np.pi * inputs[j], wires=j)

    # Start constructiong of TTN
    for j in range(N):
        qml.RY(np.pi * theta[j], wires=j)

    for j in range(0, N, 2):
        qml.CNOT(wires=[j, j + 1])
    for j in range(0, N):
        qml.RY(np.pi * theta[N + j], wires=[j])
    for j in range(1, N - 1, 2):
        qml.CNOT(wires=[j, j + 1])
    for j in range(0, N):
        qml.RY(np.pi * theta[2 * N + j], wires=[j])
    #
    # Measurement producing 16
    out_array = [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]
    return out_array

class OthelloNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        weight_shapes = {"theta": 3 * n_qubits}
        qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)
        # qlayer.build()

        print("1")
        # print(qlayer.output_shape)
        # print("2")
        # print(qlayer)
        # print(qnode)


        # Neural Net
        qlayer.build(512)
        self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y

        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(x_image)))         # batch_size  x board_x x board_y x num_channels
        # h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        # h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        # h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv1)
        # q = qlayer(h_conv4_flat)
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024
        # print("1")
        # print(s_fc2)
        # qlayer.build(512)
        s_fc2q = qlayer(s_fc2)
        # print("2")
        # print(s_fc2q)
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2q)   # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2q)                    # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))








