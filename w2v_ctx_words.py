#!/usr/bin/env python
import argparse
from collections import Counter

import numpy as np
from gensim.models import Word2Vec

import kmeans
import utils


def word_clusters(w2v, word, n_senses, min_weight=1.5, min_count=20):
    weights = utils.load_weights('../corpora/ad-nouns/cdict/', word)
    contexts = utils.load_contexts('../corpora/ad-nouns-contexts-100k', word)
    words = [
        w for w, cnt in Counter(w for ctx in contexts for w in ctx).items()
        if cnt >= min_count and weights.get(w, 0) > min_weight]
    w2v_vecs = np.array([w2v[w] for w in words if w in w2v])
    km = kmeans.KMeans(w2v_vecs, k=n_senses, metric='cosine', verbose=0)
    words = np.array(words)
    return words, weights, km


def print_senses(words, weights, km, n_senses):
    for sense in range(n_senses):
        indices = km.Xtocentre == sense
        distances = km.distances[indices]
        sense_words = words[indices]
        min_indices = np.argsort(distances)[:10]
        min_words = list(sense_words[min_indices])
        min_words.sort(key=lambda w: weights.get(w, 0), reverse=True)
        print(sense, ' '.join(min_words[:5]))


def run_all(*, model, word, n_runs, n_senses):
    w2v = Word2Vec.load(model)
    words = [word] if word else utils.all_words
    for word in words:
        print()
        print(word)
        for _ in range(n_runs):
            words, weights, km = word_clusters(w2v, word, n_senses)
            print_senses(words, weights, km, n_senses)

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
