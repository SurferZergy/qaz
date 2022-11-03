import sys
sys.path.append('..')
from utils import *

import argparse
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras.optimizers import *

import math

import tensorflow as tf
import tensorflow_quantum as tfq

import gym, cirq, sympy
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit


def one_qubit_rotation(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rx(symbols[0])(qubit),
          cirq.ry(symbols[1])(qubit),
          cirq.rz(symbols[2])(qubit)]


def entangling_layer(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops

def generate_circuit(qubits, n_layers):
    """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
    # Number of qubits
    n_qubits = len(qubits)

    # Sympy symbols for variational angles
    params = sympy.symbols(f'theta(0:{3 * (n_layers + 1) * n_qubits})')
    params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))

    # Sympy symbols for encoding angles
    inputs = sympy.symbols(f'x(0:{n_layers})' + f'_(0:{n_qubits})')
    inputs = np.asarray(inputs).reshape((n_layers, n_qubits))

    # Define circuit
    circuit = cirq.Circuit()
    for l in range(n_layers):
        # Variational layer
        circuit += cirq.Circuit(one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
        circuit += entangling_layer(qubits)
        # Encoding layer
        circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))

    # Last varitional layer
    circuit += cirq.Circuit(one_qubit_rotation(q, params[n_layers, i]) for i, q in enumerate(qubits))

    return circuit, list(params.flat), list(inputs.flat)

class ReUploadingPQC(tf.keras.layers.Layer):
    """
    Performs the transformation (s_1, ..., s_d) -> (theta_1, ..., theta_N, lmbd[1][1]s_1, ..., lmbd[1][M]s_1,
        ......., lmbd[d][1]s_d, ..., lmbd[d][M]s_d) for d=input_dim, N=theta_dim and M=n_layers.
    An activation function from tf.keras.activations, specified by `activation` ('linear' by default) is
        then applied to all lmbd[i][j]s_i.
    All angles are finally permuted to follow the alphabetical order of their symbol names, as processed
        by the ControlledPQC.
    """

    def __init__(self, qubits, n_layers, observables, activation="linear", name="re-uploading_PQC"):
        super(ReUploadingPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)

        circuit, theta_symbols, input_symbols = generate_circuit(qubits, n_layers)

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
        initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
        trainable=True, name="thetas"
        )

        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers,))
        self.lmbd = tf.Variable(
        initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
        )

        # Define explicit symbol order.
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)

    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])

class Alternating(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(Alternating, self).__init__()
        self.w = tf.Variable(
            initial_value=tf.constant([[(-1.) ** i for i in range(output_dim)]]), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

def generate_model_policy(qubits, n_layers, n_actions, beta, observables, observables2):
    """Generates a Keras model for a data re-uploading PQC policy."""

    input_tensor = tf.keras.Input(shape=len(qubits), dtype=tf.dtypes.float32, name='input')
    print('obs')
    print(observables)
    print(observables2)
    re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables, name='pi_re-upload_PQC')([input_tensor])
    process = tf.keras.Sequential([
        # Alternating(n_actions), #Try #1
        # tf.keras.layers.Lambda(lambda x: x * beta),
        # tf.keras.layers.Softmax()
    ], name="observables-policy")
    pi_q = process(re_uploading_pqc)
    # pi = tf.keras.layers.Dense(n_actions, activation='softmax', name='pi')(pi_q) #try #1
    # pi = tf.keras.layers.Dense(n_actions, activation='linear', name='pi')(pi_q) #try #2 - seems to mask all actions
    pi = tf.keras.layers.Dense(n_actions, activation='sigmoid', name='pi')(pi_q) #try #1

    # re_uploading_pqc_v = ReUploadingPQC(qubits, n_layers, cirq.Z(cirq.GridQubit(0, 0)), activation='tanh', name='value_re-upload_PQC')([input_tensor])
    re_uploading_pqc_v = ReUploadingPQC(qubits, n_layers, observables2, activation='tanh', name='value_re-upload_PQC')([input_tensor])

    process_v = tf.keras.Sequential([
        # Alternating(1), #Try #1
        # tf.keras.layers.Lambda(lambda x: x * beta),
        # tf.keras.layers.Activation(tf.nn.tanh)
    ], name="observables-v")
    v_q = process_v(re_uploading_pqc_v)
    # v = tf.keras.layers.Dense(1, activation='tanh', name='v')(v_q) #try #1
    v = tf.keras.layers.Dense(1, activation='linear', name='v')(v_q) #try #2 -seems to mask all actions
    #3.1 try linear again since tanh already once

    model = tf.keras.Model(inputs=[input_tensor], outputs=[pi, v])

    return model, pi, v



class OthelloNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        self.n_qubits = 16
        self.n_layers = 5  # Number of layers in the PQC
        self.qubits = cirq.GridQubit.rect(1, self.n_qubits)
        # self.ops = [cirq.Z(q) for q in self.qubits]
        self.ops = [cirq.Z(q) for q in self.qubits[:15]]#try 15-1 obs
        self.ops2 = [cirq.Z(q) for q in self.qubits[15:]]#try 15-1 obs
        print('ops')
        print(self.ops)
        print(self.ops2)
        self.observables = [reduce((lambda x, y: x * y), self.ops)]  # Z_0*Z_1*Z_2*Z_3
        self.observables2 = [reduce((lambda x, y: x * y), self.ops2)] # try 8-8
        # self.model, self.pi, self.v = generate_model_policy(self.qubits, self.n_layers, self.action_size, 1.0, self.observables)
        self.model, self.pi, self.v = generate_model_policy(self.qubits, self.n_layers, self.action_size, 1.0, self.observables, self.observables2)
        # self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=tf.keras.optimizers.Adam(args.lr))
        self.model.compile(loss=['binary_crossentropy', tf.keras.losses.Huber()], optimizer=tf.keras.optimizers.Adam(args.lr)) #try #3



        # Neural Net
        # self.input_boards = Input(shape=(self.board_x, self.board_y))    # s: batch_size x board_x x board_y
        #
        # x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)                # batch_size  x board_x x board_y x 1
        # h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(x_image)))         # batch_size  x board_x x board_y x num_channels
        # h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='same', use_bias=False)(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        # h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        # h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        # h_conv4_flat = Flatten()(h_conv4)
        # s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        # s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024
        # self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)   # batch_size x self.action_size
        # self.v = Dense(1, activation='tanh', name='v')(s_fc2)                    # batch_size x 1
        #
        # self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        # self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(args.lr))
