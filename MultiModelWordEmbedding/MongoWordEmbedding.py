# -*- coding: utf8 -*-

import pymongo
import gensim.models
import fasttext
import fasttext.util
import pandas as pd

import os
import json
import numpy as np

from tqdm import tqdm

def download_embedding_models(embedding_path):

    def get_or_create_collection(collection):
        if collection in db.collection_names():
            return db[collection], False

        db.create_collection(collection)
        return db[collection], True

    db = pymongo.MongoClient()['embedding']
    step_size = 100
    
    model_name = 'Word2Vec'
    print(f'Load {model_name} embedding into MongoDB')
    collection, collection_created = get_or_create_collection(model_name)

    if collection_created:

        print(f'Loading {model_name} model')
        model = gensim.models.KeyedVectors.load_word2vec_format(
            os.path.join(embedding_path, 'word2vec/GoogleNews-vectors-negative300.bin'),
            binary=True
        )

        print(f'Saving {model_name} embeddings in data base')
        
        idxs = list(model.key_to_index.values())
        total_words = len(idxs)
        for i in tqdm(range(0, total_words, step_size)):
            current_idxs = idxs[i: min(total_words-1,i+step_size)]
            words = [model.index_to_key[i] for i in current_idxs]
            vectors = model[current_idxs]
            collection.insert_many([
                {'word': word, 'vector': vector}
                for word, vector
                in zip(words, vectors)
            ])

    # Load Fasttext

    model_name = 'FastText'
    print(f'Load {model_name} embedding into MongoDB')
    collection, collection_created = get_or_create_collection(model_name)

    if collection_created:
        print(f'Loading {model_name} model')
        model = fasttext.load_model(
            os.path.join(embedding_path, 'fasttext/cc.en.300.bin')
        )

        print(f'Saving {model_name} embeddings in data base')
        total_words = len(model.get_words())
        for i in tqdm(range(0, total_words, step_size)):
            words = model.get_words()[i: min(total_words-1,i+step_size)]
            vectors = [model.get_word_vector(w) for w in words]
            collection.insert_many([
                {'word': word, 'vector': vector} 
                for word, vector 
                in zip(words, vectors)
            ])

    # Load GloVe
    model_name = 'GloVe'
    print(f'Load {model_name} embedding into MongoDB')
    collection, collection_created = get_or_create_collection(model_name)

    if collection_created:
        print(f'Loading {model_name} model')
        model = pd.read_csv(
            os.path.join(embedding_path, 'glove/glove.840B.300d.txt'),
            sep=' ',
            quoting=3,
            header=None, 
            index_col=0
        )

        print(f'Saving {model_name} embeddings in data base')
        total_words = len(len(model))
        for i in tqdm(range(0, total_words, step_size)):
            words = model.iloc[i: min(total_words-1,i+step_size)].index.to_list()
            vectors = model.iloc[i: min(total_words-1,i+step_size)].values.to_list()
            collection.insert_many([
                {'word': word, 'vector': vector} 
                for word, vector 
                in zip(words, vectors)
            ])

    # Load LSA
    model_name = 'LSA'
    print(f'Load {model_name} embedding into MongoDB')
    collection, collection_created = get_or_create_collection(model_name)

    if collection_created:
        model = pd.DataFrame(
            np.load(os.path.join(embedding_path, 'LSA/tasa_300/matrix.npy')), 
            index=json.load(open(os.path.join(embedding_path, 'LSA/tasa_300/dictionary.json'), 'r'))
        )

        print(f'Saving {model_name} embeddings in data base')
        total_words = len(len(model))
        for i in tqdm(range(0, total_words, step_size)):
            words = model.iloc[i: min(total_words-1,i+step_size)].index.to_list()
            vectors = model.iloc[i: min(total_words-1,i+step_size)].values.to_list()
            collection.insert_many([
                {'word': word, 'vector': vector} 
                for word, vector 
                in zip(words, vectors)
            ])

class MongoWordEmbedding():

    def __init__(self, model_name):
        self.db = pymongo.MongoClient(**settings['mongo_client'])['embedding']
        if model_name not in self.db.collection_names():
            raise LookupError(f'The {model_name} is not yet loaded in MongoDB, run:\n\nfrom MongoWordEmbedding import download_embedding_models\ndownload_embedding_models()\n')

        self.model_name = model_name
        self.embedding = self.db[self.model_name]

    def __getitem__(self, word):
        return self.embedding.find_one({'word': word})['vector']

    def get_vocab(self):
        return [d['word'] for d in self.embedding.find()]
    
    def get_vocab_vectors(self):
        return [document['vector'] for document in self.embedding.find({})]