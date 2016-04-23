#!/usr/bin/env python
import argparse
from collections import defaultdict, Counter

import numpy as np
from gensim.models import Word2Vec

import kmeans
import utils


def word_clusters_neighbours(w2v, word, n_senses, *, window):
    assert window
    similar = w2v.most_similar(positive=[word], topn=100)
    words = np.array([w for w, _ in similar])
    word_vectors = np.array([w2v[w] for w in words])
    km = kmeans.KMeans(word_vectors, k=n_senses, metric='cosine', verbose=0)
    return words, km

word_clusters_neighbours.threshold = 0.75


def word_clusters_ctx(w2v, word, n_senses, min_weight=1.5, min_count=20, *,
                      window):
    weights, contexts = utils.weights_contexts(word, window)
    words = [
        w for w, cnt in Counter(w for ctx in contexts for w in ctx).items()
        if cnt >= min_count and weights.get(w, 0) > min_weight and w in w2v]
    w2v_vecs = np.array([w2v[w] for w in words])
    km = kmeans.KMeans(w2v_vecs, k=n_senses, metric='cosine', verbose=0)
    words = np.array(words)
    return words, km

word_clusters_ctx.threshold = 0.50


def print_senses(w2v, sense_words, topn=5):
    for sense_id, words in sorted(sense_words.items()):
        words.sort(key=lambda w: w2v.vocab[w].count, reverse=True)
        print(sense_id, ' '.join(words[:topn]))


def run_all(*, clustering, model, word, n_runs, n_senses, window):
    clustering_fn = globals()['word_clusters_' + clustering]
    w2v = Word2Vec.load(model)
    words = [word] if word else utils.all_words
    for word in words:
        print()
        print(word)
        for _ in range(n_runs):
            words, km = clustering_fn(w2v, word, n_senses, window=window)
            sense_words = {
                sense_id: list(words[km.Xtocentre == sense_id])
                for sense_id in range(n_senses)}
            print_senses(w2v, sense_words)
            utils.print_cluster_sim(km.centres)
            mapping = utils.merge_clusters(
                km.centres, threshold=clustering_fn.threshold)
            print(mapping)
            merged_sense_words = defaultdict(list)
            for sense_id, words in sense_words.items():
                merged_sense_words[mapping[sense_id]].extend(words)
            print_senses(w2v, merged_sense_words)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('clustering', choices=['ctx', 'neighbours'])
    arg('--n-senses', type=int, default=6)
    arg('--n-runs', type=int, default=1)
    arg('--model', default='model.pkl')
    arg('--window', type=int, default=10)
    arg('--word')
    params = vars(parser.parse_args())
    print(params)
    run_all(**params)


if __name__ == '__main__':
    main()
