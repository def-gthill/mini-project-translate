import numpy as np
from sklearn import base
from tensorflow import keras

class Word2VecTransformer(base.TransformerMixin):
    def __init__(self, wv):
        self.wv = wv
    
    def fit(self, sents):
        return self
    
    def transform(self, sents):
        return [
            [
                self.wv.get_vector(word)
                for word in sent
                if word in self.wv
            ]
            for sent in sents
        ]
    
    def inverse_transform(self, vecs):
        return [
            self._inverse_transform_sentence(sent_vecs)
            for sent_vecs in vecs
        ]
    
    def _inverse_transform_sentence(self, sent_vecs):
        result = []
        for word_vec in sent_vecs:
            if np.linalg.norm(word_vec) < 0.05:
                break
            result.append(self.wv.similar_by_vector(word_vec, topn=1)[0][0])
        return result

    
def pad(vecs):
    return keras.preprocessing.sequence.pad_sequences(
        vecs, maxlen=40, dtype='float', padding='post',
        value=np.zeros(100)
    )