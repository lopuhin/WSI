#!/usr/bin/env python
import argparse

import numpy as np
from gensim.models import Word2Vec

import kmeans
import utils


def word_clusters(w2v, word, n_senses):
    similar = w2v.most_similar(positive=[word], topn=100)
    words = np.array([w for w, _ in similar])
    word_vectors = np.array([w2v[w] for w in words])
    km = kmeans.KMeans(word_vectors, k=n_senses, metric='cosine', verbose=0)
    return words, km


def print_senses(w2v, words, km, n_senses, topn=5):
    for sense in range(n_senses):
        sense_words = list(words[km.Xtocentre == sense])
        sense_words.sort(key=lambda w: w2v.vocab[w].count, reverse=True)
        print(sense, ' '.join(sense_words[:topn]))


def run_all(*, model, word, n_runs, n_senses):
    w2v = Word2Vec.load(model)
    words = [word] if word else utils.all_words
    for word in words:
        print()
        print(word)
        for _ in range(n_runs):
            words, km = word_clusters(w2v, word, n_senses)
            print_senses(w2v, words, km, n_senses)
            utils.print_cluster_sim(km.centres)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--n-senses', type=int, default=6)
    arg('--n-runs', type=int, default=1)
    arg('--model', default='model.pkl')
    arg('--word')
    params = vars(parser.parse_args())
    print(params)
    run_all(**params)


if __name__ == '__main__':
    main()
