"""
Functions for machine learning
"""

import pickle
import traceback
import pathlib
import os.path


def load_or_train(path, training_function, method='pickle'):
    """
    Tries to load a pickled trained model from the specified
    path. Returns this model if the file is present. Otherwise,
    trains a new model by calling the specified training
    function (with no arguments), saves the resulting
    model to the path, and returns the model.
    
    Supported methods: 'pickle', 'tensorflow', 'word2vec', 'keyedvectors'
    """
    try:
        return load(path, method=method)
    except OSError:
        trained_model = training_function()    
        try:
            pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
            save(path, trained_model, method=method)
        except:
            traceback.print_exc()
        return trained_model


def load(path, method='pickle'):
    """
    Loads a model from the specified path.
    
    Supported methods: 'pickle', 'tensorflow', 'word2vec', 'keyedvectors'
    """
    return {
        'pickle': _load_pickle,
        'tensorflow': _load_tensorflow,
        'word2vec': _load_word2vec,
        'keyedvectors': _load_keyedvectors,
    }[method](path)


def save(path, model, method='pickle'):
    """
    Saves a model to the specified path.
    
    Supported methods: 'pickle', 'tensorflow', 'word2vec', 'keyedvectors'
    """
    {
        'pickle': _save_pickle,
        'tensorflow': _save_tensorflow,
        'word2vec': _save_word2vec,
        'keyedvectors': _save_keyedvectors,
    }[method](path, model)


def _load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    

def _save_pickle(path, model):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def _load_tensorflow(path):
    from tensorflow import keras
    return keras.models.load_model(path)


def _save_tensorflow(path, model):
    model.save(path)
    

def _load_word2vec(path):
    import gensim
    return gensim.models.Word2Vec.load(path)
    

def _save_word2vec(path, model):
    model.save(path)


def _load_keyedvectors(path):
    import gensim
    return gensim.models.KeyedVectors.load(path)


def _save_keyedvectors(path, model):
    model.save(path)
