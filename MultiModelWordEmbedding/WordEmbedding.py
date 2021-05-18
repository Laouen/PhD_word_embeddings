import gensim.models
import fasttext
import fasttext.util
import pandas as pd
import numpy as np

import json
import os
import glob

from MultiModelWordEmbedding import settings


def get_embedding(suffix):
    return os.path.join(settings.EMBEDDINGS_FODLER, suffix)

class Word2Vec():

    def __init__(self):
        self.model_name = 'Word2Vec'
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            get_embedding('word2vec/GoogleNews-vectors-negative300.bin'),
            binary=True
        )
    
    def __getitem__(self, words):
        return self.model[words]
    
    def get_vocab(self):
        return set(self.model.index_to_key)


class FastText():
    
    def __init__(self):
        self.model_name = 'FastText'
        self.model = fasttext.load_model(get_embedding('fasttext/cc.en.300.bin'))

    def __getitem__(self, words):
        if type(words) == str:
            return self.model.get_word_vector(words)
        
        return np.array([
            self.model.get_word_vector(w)
            for w in words
        ])
    
    def get_vocab(self):
        return set(self.model.get_words())


class GloVe():
    
    def __init__(self):
        self.model_name = 'GloVe'
        self.model = pd.read_csv(
            get_embedding('glove/glove.840B.300d.txt'), 
            sep=' ',
            quoting=3,
            header=None, 
            index_col=0
        )
        
    def __getitem__(self, words):
        return self.model.loc[words].to_numpy()
    
    def get_vocab(self):
        return set(self.model.index.to_list())


class LSA():
    
    def __init__(self):
        self.model_name = 'LSA'
        self.model = pd.DataFrame(
            np.load(get_embedding('LSA/tasa_300/matrix.npy')), 
            index=json.load(open(get_embedding('LSA/tasa_300/dictionary.json'), 'r'))
        )
        
    def __getitem__(self, words):
        return self.model.loc[words].to_numpy()
    
    def get_vocab(self):
        return set(self.model.index.to_list())
    

class MultilingualFastText():
    def __init__(self):
        self.model_name = 'Multilingual FastText'
        self.models = {
            file.split('.')[1]: gensim.models.KeyedVectors.load_word2vec_format(file)
            for file in glob.glob(get_embedding('fasttext/aligned/wiki.*.align.vec'))
        }

    def __getitem__(self, words_lang):
        words, lang = words_lang
        return self.models[lang][words]
    
    def get_vocab(self):
        return set(self.model.index_to_key)


MODELS = {
    'Word2Vec': Word2Vec,
    'GloVe': GloVe,
    'FastText': FastText,
    'LSA': LSA,
    'Multilingual FastText': MultilingualFastText
}

WORD_EMBEDDINGS = list(MODELS.keys())

class WordEmbedding():

    def __init__(self, model):
        self.model = MODELS[model]()
    
    def __getitem__(self, words):
        return self.model[words]
    
    def get_vocab(self):
        return self.model.get_vocab()
    
    def get_vocab_vectors(self):
        return [self.__getitem__(word) for word in self.get_vocab()]