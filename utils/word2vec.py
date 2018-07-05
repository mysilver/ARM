import numpy
from gensim.models.keyedvectors import KeyedVectors
from scipy import spatial
import heapq

#model = KeyedVectors.load_word2vec_format(
#    'D:\data\GoogleNews-vectors-negative300.bin', binary=True, limit=200000)


def word_vector(word):
    if word in model.vocab:
        return model.word_vec(word)

    return None


def find_most_similar(vec, vectors_dictionary, topn=100):
    h = []
    counter = 0
    with open(vectors_dictionary, 'rt') as f:
        for line in f.readlines():
            # counter +=1
            line = line.split("\t")
            id = line[0]
            v = line[1:]
            similarity = 1 - spatial.distance.cosine([float(i) for i in vec], [float(i) for i in v])
            if len(h) < topn:
                heapq.heappush(h, (id, similarity))
            else:
                heapq.heappushpop(h, (id, similarity))


    return h
