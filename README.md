# PhD_word_embeddings
This repository holds two simple class wrapper implementing some word embedding models.

1. MongoWordEmbedding implements a MongoDB based wrapper that consume embeddings from a mongoDB database, this is memory efficient but require a MongoDB instance.
2. WordEmbedding implements an in memory wrapper that loads the models into the memory, this is memory inefficient but do not require a MongoDB instance.

# Current used pretrained word embedding models

1. Word2Vec: TODO insert link
2. FastText (en): TODO insert link
3. FastText (Aligned): https://fasttext.cc/docs/en/aligned-vectors.html
    a. Spanish: https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.es.align.vec
    b. English: https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec
    c. Portugese: https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.pt.align.vec
4. LSA:


# Requirements
The MongoWordEmbedding uses MongoDB, in order to use that version a running conextion to a MongoDB with read/write permisions is required

## Install in Linux
0. Install MongoDB if you intend to use MongoWordEmbedding.
1. Clone this repo.
2. Enter to the repo root dir from a console.
3. python setup.py install.
4. Configure settings.json.

## Settings.json

The settings.json file stores the following configuration:
```json
{
    'embeddings_folder': path to the folder where the embedding model files are stored with the structire shown in (1)
    'mongo_client': a dictionary with the parameters of the Pymongo.MongoClient(**parameters) database conexion method
}
```

(1) The word embedding folder and file structure is the following:
```
word_mebddings_folder
|
├── word2vec
|   └── GoogleNews-vectors-negative300.bin
|
├── fasttext
|   └── cc.en.300.bin
|
├── glove
|   └── glove.840B.300d.txt
|
└── LSA
    └── tasa_300
        └── matrix.npy
```

## Examples using the WordEmbedding module (without Mongo)
```python
from MultiModelWordEmbedding.WordEmbedding import WordEmbedding

w2v = WordEmbedding('Word2Vec')
w2v['cat']
```

## Examples using the MongoWordEmbedding module (with Mongo)

```python
from MultiModelWordEmbedding.MongoWordEmbedding import download_embedding_models, MongoWordEmbedding
download_embedding_models('words_embedding_folder')

w2v = MongoWordEmbedding('Word2Vec')
w2v['cat']
```