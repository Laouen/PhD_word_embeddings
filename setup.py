from setuptools import setup

setup(
    name='MongoWordEmbeddings',
    version='1.0',
    description='Package for accessing word embedding models from MongoDB.',
    author_email='laouen.belloli@gmail.com',
    package=['MultiModelWordEmbedding'],
    setup_requires=["numpy"], 
    install_requires=[
        'pymongo',
        'gensim',
        'fasttext',
        'tqdm',
        'numpy',
        'pandas',
    ]
)