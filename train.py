#!/usr/bin/env python3

"""
Example training program for b-tagging with keras
to run: python train.py train_file test_file
"""

import h5py
from argparse import ArgumentParser
import numpy as np
from itertools import cycle

import keras
from keras import layers
from keras.models import Model
from keras.callbacks import ModelCheckpoint,History

import matplotlib.pyplot as plt

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('train_file')
    parser.add_argument('test_file')
    return parser.parse_args()

def flatten(ds):
    """
    Returns a flattened NumPy array

    Args:
        ds (structured NumPy array): A structured n dimensional array
    Returns:
        flat (non-structured NumPy array): A "flattened" array of floats with
                     additional dimension to represent fields in the structure
    """
    ftype = [(n, float) for n in ds.dtype.names]
    flat = ds.astype(ftype).view(float).reshape(ds.shape + (-1,))
    return flat

def generate(input_file,jet_vars,trk_vars,batch_size=100):
    """
    Returns a generator for a keras model
    (either training or evaluation)

    Args:
        input_file (str): path to your chosen input file
        jet_vars (list of strs): list of jet vars to return
        trk_vars (list of strs): list of track vars to return
        batch_size (int): number of samples per batch
    Returns:
        tuple of flattened numpy arrays with track vars,
        jet vars, one-hot flavor labels, and mv2c10 value
    

    """
    #open your hdf5 file
    with h5py.File(input_file, 'r') as hdf_file:
        #count the # of jets
        n_jets = hdf_file['jets'].shape[0]
        #avoid partial final batch
        limit = int(n_jets / batch_size) * batch_size
        all_jets = hdf_file['jets']
        all_tracks = hdf_file['tracks']
        #batch it up!
        for start_index in cycle(range(0, limit, batch_size)):
            sl = slice(start_index,start_index + batch_size)
            jets = all_jets[sl]
            tracks = all_tracks[sl,:]

            fl_jets = flatten(jets[jet_vars])
            fl_trks = flatten(tracks[trk_vars])
            labels = jets['LabDr_HadF']
            charge = jets['mv2c10']
            one_hot = np.vstack([labels == n for n in [0, 4, 5, 15]]).T
            yield [fl_trks, fl_jets], [one_hot, charge]

def train(train_file,jet_vars,trk_vars):
    """
    Train the model

    Args:
        train_file (str): path to the training hd5 model
        jet_vars (list of strs): list of jet vars to be generated
        trk_vars (list of strs): list of track vars to be generated

    Returns:
        keras History callback, trained keras model
    """
    print(type(jet_vars))
    model = get_model(len(jet_vars), len(trk_vars))
    #checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True, monitor='loss')
    train_history = model.fit_generator(generate(train_file,jet_vars,trk_vars), steps_per_epoch=500, callbacks=[],epochs=5)
    
    return train_history, model   
 
def get_model(n_jet_vars, n_trk_var):
    """
    Make the model

    Args:
        n_vx_vars (int): The number of jet variables
        n_trk_var (int): The number of track variables
    Returns:
        model (Keras Model): The model
    """
    # setup inputs
    jets = layers.Input(shape=(n_jet_vars,), name='jets')
    tracks = layers.Input(shape=(60, n_trk_var), name='tracks')

    # add GRU to process tracks
    gru = layers.GRU(5)(tracks)

    # merge with the jet inputs and feed to a dense layer
    merged = layers.concatenate([gru, jets])
    dense = layers.Dense(10, activation='relu')(merged)

    # add flavors output
    flavor = layers.Dense(4, activation='softmax', name='flavor')(dense)

    # add charge output
    charge = layers.Dense(1, name='charge')(dense)

    # build and compile the model
    model = Model(inputs=[tracks, jets], outputs=[flavor, charge])
    model.compile(optimizer='adam',
                  loss=['categorical_crossentropy', 'mean_squared_error'],
                  metrics=['accuracy', 'accuracy'])
    return model

if __name__ == '__main__':
    args = get_args()
    jet_vars = ['pt', 'eta']
    trk_vars = ['d0', 'charge']
    #i think you need this as a placeholder...
    train_history = History()
    train_history, model = train(args.train_file,jet_vars,trk_vars)
    model.evaluate_generator(generate(args.test_file,jet_vars,trk_vars),10) 


