import numpy
from gensim.models.keyedvectors import KeyedVectors
from scipy import spatial
import heapq

model = KeyedVectors.load_word2vec_format(
    'D:\data\GoogleNews-vectors-negative300.bin', binary=True, limit=200000)


def word_vector(word):
    if word in model.vocab:
        return model.word_vec(word)

    return None


def find_most_similar(vec, vectors_dictionary, topn=100):
    h = []
    for id, v in vectors_dictionary.items():
        similarity = 1 - spatial.distance.cosine(vec, v)
        if len(h) < topn:
            heapq.heappush(h, (id, similarity))
        else:
            heapq.heappushpop(h)

    return h
