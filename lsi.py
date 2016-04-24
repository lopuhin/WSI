#!/usr/bin/env python
import logging
import argparse
import random
import os.path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from gensim import corpora
from gensim.models.lsimodel import LsiModel
import rl_wsd_labeled
from sklearn.metrics import v_measure_score, adjusted_rand_score

import utils


def word_lsi(word, num_topics, window, limit=None, min_weight=1.0):
    weights, contexts = utils.weights_contexts(word, window)
    weights_flt = partial(utils.weights_flt, weights, min_weight)
    contexts = [ctx for ctx in map(weights_flt, contexts) if ctx]
    random.shuffle(contexts)
    if limit:
        contexts = contexts[:limit]
    print(len(contexts))
    dictionary = corpora.Dictionary(contexts)
    corpus = [dictionary.doc2bow(ctx) for ctx in contexts]
    lsi = LsiModel(
        corpus, id2word=dictionary, num_topics=num_topics)
    return lsi


def run_all(*, word, n_runs, limit, n_senses, window):
    words = [word] if word else utils.all_words
    futures = []
    with ProcessPoolExecutor(max_workers=4) as e:
        for word in words:
            futures.extend(
                (word, e.submit(
                    word_lsi, word, n_senses, limit=limit, window=window))
                for _ in range(n_runs))
    results_by_word = defaultdict(list)
    for word, f in futures:
        results_by_word[word].append(f.result())
    for word, results in sorted(results_by_word.items()):
        print()
        print(word)
        for lsi in results:
            sense_words = {sense_id: lsi.show_topic(sense_id)
                           for sense_id in range(lsi.num_topics)}
            utils.print_senses(sense_words)


def main():
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--n-senses', type=int, default=6)
    arg('--limit', type=int)
    arg('--n-runs', type=int, default=1)
    arg('--window', type=int, default=10)
    arg('--word')
    params = vars(parser.parse_args())
    print(params)
    run_all(**params)


if __name__ == '__main__':
    main()
