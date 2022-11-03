# -*- coding: utf-8 -*-
"""
Created on Mon May  2 12:06:31 2022

This is a test bed for the newly created layer we made in tensorflow

@author: scrowe
"""


from pennylane import numpy as np
#import numpy as np
import tensorflow as tf
import time
from tensorflow import keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pennylane as qml
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/scrowe/Desktop/TTN_Quanvolve/')
#from TTN_Quanvolve_layer import TTN_Quanvolver

#Set seeds for random number generator
np.random.seed(1)           # Seed for NumPy random number generator
tf.random.set_seed(1)

#Set number of epochs
n_epochs = 50   # Number of optimization epochs


#%% Enter classical image directories
train_image_file = r'C:\Users\scrowe\Desktop\Quan_Extraction_CNN\traffic-dataset\training'
test_image_file = r'C:\Users\scrowe\Desktop\Quan_Extraction_CNN\traffic-dataset\testing'

#%% Get the training data
imdim=64

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory(train_image_file,
                                                 target_size = (imdim, imdim),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


#%% Get test set
test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


test_set = test_datagen.flow_from_directory(test_image_file,
                                            target_size = (imdim, imdim),
                                            batch_size = 4,
                                            class_mode = 'binary')

#%% Define the quantum model

n_qubits=12
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs,theta):

    N=n_qubits

    # Encoding of 16 dim array
    for j in range(N):
        qml.RY(np.pi * inputs[j], wires=j)

    #Start constructiong of TTN
    for j in range(N):
        qml.RY(np.pi*theta[j],wires=j)

    for j in range(0,N,2):
        qml.CNOT(wires=[j,j+1])
    for j in range(0,N):
        qml.RY(np.pi*theta[N+j],wires=[j])
    for j in range(1,N-1,2):
        qml.CNOT(wires=[j,j+1])
    for j in range(0,N):
        qml.RY(np.pi*theta[2*N+j],wires=[j])
#
    # Measurement producing 16
    out_array=[qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]
    return out_array

weight_shapes={"theta": 3*n_qubits}
qlayer=qml.qnn.KerasLayer(qnode,weight_shapes,output_dim=n_qubits)
################################################


#dev = qml.device("default.qubit", wires=4)
# Random circuit parameters
#rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))


#def circuit(phi):
#    # Encoding of 4 classical input values
#    for j in range(4):
#        qml.RY(np.pi * phi[j], wires=j)
#
#    # Random quantum circuit
#    qml.RandomLayers(rand_params, wires=list(range(4)))
#
#    # Measurement producing 4 classical output values
#    return [qml.expval(qml.PauliZ(j)) for j in range(4)]
#
#    # Measurement producing 4 classical output values
#    return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]






#%% Define the model
opt = tf.keras.optimizers.SGD(learning_rate=0.5)
def MyModel_quantum():
    """Initializes and returns a custom Keras model
    which is ready to be trained."""
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',input_shape=[imdim, imdim, 3]),
        keras.layers.MaxPool2D(pool_size=2, strides=2),
        keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        keras.layers.MaxPool2D(pool_size=2, strides=2),
        keras.layers.Flatten(),
        #qlayer,
        keras.layers.Dense(units=128, activation='sigmoid'),
        #keras.layers.Dense(units=8, activation='sigmoid'),
        qlayer,
        keras.layers.Dense(units=1, activation='sigmoid')
        # keras.layers.Flatten(),
        # keras.layers.Dense(10, activation="softmax"),
        # keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def MyModel_classical():
    """Initializes and returns a custom Keras model
    which is ready to be trained."""
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',input_shape=[imdim, imdim, 3]),
        keras.layers.MaxPool2D(pool_size=2, strides=2),
        keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        keras.layers.MaxPool2D(pool_size=2, strides=2),
        keras.layers.Flatten(),
        #qlayer,
        keras.layers.Dense(units=128, activation='sigmoid'),
        #keras.layers.Dense(units=8, activation='sigmoid'),
        #qlayer,
        keras.layers.Dense(units=1, activation='sigmoid')
        # keras.layers.Flatten(),
        # keras.layers.Dense(10, activation="softmax"),
        # keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def MyModel_Vanilla():
    """Initializes and returns a custom Keras model
    which is ready to be trained."""
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu',input_shape=[imdim, imdim, 3]),
        keras.layers.MaxPool2D(pool_size=2, strides=2),
        keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        keras.layers.MaxPool2D(pool_size=2, strides=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation='sigmoid'),
        keras.layers.Dense(units=1, activation='sigmoid')
        # keras.layers.Flatten(),
        # keras.layers.Dense(10, activation="softmax"),
        # keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss="mae",
        run_eagerly=True
    )
    return model


#%%
qmodel = MyModel_quantum()
cmodel = MyModel_classical()

start=time.time()
qhistory = qmodel.fit(
    training_set,
    validation_data=test_set,
    epochs=n_epochs,
    batch_size=8#,
    #verbose=2
)
stop=time.time()

print(f"Training time: {stop - start}s")

start=time.time()
chistory = cmodel.fit(
    training_set,
    validation_data=test_set,
    epochs=n_epochs,
    batch_size=8#,
    #verbose=2
)
stop=time.time()

print(f"Training time: {stop - start}s")




#%%%%%%



#Plot the results
plt.style.use("seaborn")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

ax1.plot(chistory.history["val_accuracy"], "-og",label='Classical')
ax1.plot(qhistory.history["val_accuracy"], "-ob",label='Quantum layer')
ax1.set_ylabel("Accuracy")
ax1.set_ylim([0, 1])
ax1.legend()
ax1.set_xlabel("Epoch")
#ax1.legend()

ax2.plot(chistory.history["val_loss"], "-og",label='Classical')
ax2.plot(qhistory.history["val_loss"], "-ob",label='Quantum layer')
ax2.set_ylabel("Loss")
ax2.set_ylim(top=1,bottom=0)
ax2.set_xlabel("Epoch")
ax2.legend()
plt.suptitle('Validation loss and accuracy functions (Classical v Quantum) ',size=18,fontweight='bold')
plt.tight_layout()
plt.show()
