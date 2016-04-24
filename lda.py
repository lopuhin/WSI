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
# from gensim.models.ldamulticore import LdaMulticore as LdaModel
# from gensim.models import HdpModel
from gensim.models import LdaModel
import rl_wsd_labeled
from sklearn.metrics import v_measure_score, adjusted_rand_score

import utils


def word_lda(word, num_topics, window, limit=None, min_weight=1.0):
    weights, contexts = utils.weights_contexts(word, window)
    weights_flt = partial(_weights_flt, weights, min_weight)
    contexts = [ctx for ctx in map(weights_flt, contexts) if ctx]
    random.shuffle(contexts)
    if limit:
        contexts = contexts[:limit]
    print(len(contexts))
    dictionary = corpora.Dictionary(contexts)
    corpus = [dictionary.doc2bow(ctx) for ctx in contexts]
    lda = LdaModel(
        corpus, id2word=dictionary, num_topics=num_topics,
        passes=4, iterations=100, alpha='auto')
    return lda, dictionary, weights_flt


def _weights_flt(weights, min_weight, ctx):
    return [w for w in ctx if weights.get(w, 0) > min_weight]


def get_scores(lda, dictionary, word, weights_flt):
    labeled_fname = rl_wsd_labeled.contexts_filename('nouns', 'RuTenTen', word)
    if os.path.exists(labeled_fname):
        _senses, contexts = rl_wsd_labeled.get_contexts(labeled_fname)
        documents = [dictionary.doc2bow(weights_flt(utils.normalize(ctx)))
                     for ctx, _ in contexts]
        gamma, _ = lda.inference(documents)
        pred_topics = gamma.argmax(axis=1)
        true_labels = np.array([int(ans) for _, ans in contexts])

        ari = adjusted_rand_score(true_labels, pred_topics)
        v_score = v_measure_score(true_labels, pred_topics)
        return ari, v_score


def print_topics(lda, dictionary, topn=5):
    for topic_id in range(lda.num_topics):
        terms = lda.get_topic_terms(topic_id, topn=topn)
        print(topic_id, ' '.join(dictionary[wid] for wid, _ in terms))


def print_cluster_sim(lda, dictionary):
    topics = []
    for topic_id in range(lda.num_topics):
        topic = np.zeros(len(dictionary))
        topics.append(topic)
        for idx, v in lda.get_topic_terms(topic_id, topn=len(dictionary)):
            topic[idx] = v
    utils.print_cluster_sim(np.array(topics))


def run_all(*, word, n_runs, limit, n_senses, window):
    words = [word] if word else utils.all_words
    futures = []
    with ProcessPoolExecutor(max_workers=4) as e:
        for word in words:
            futures.extend(
                (word, e.submit(
                    word_lda, word, n_senses, limit=limit, window=window))
                for _ in range(n_runs))
    results_by_word = defaultdict(list)
    for word, f in futures:
        results_by_word[word].append(f.result())
    aris, v_scores = [], []
    for word, results in results_by_word.items():
        print()
        print(word)
        word_aris, word_v_scores = [], []
        for lda, dictionary, weights_flt in results:
            print_topics(lda, dictionary)
            print_cluster_sim(lda, dictionary)
            scores = get_scores(lda, dictionary, word, weights_flt)
            if scores:
                ari, v_score = scores
                print('ARI: {:.3f}, V-score: {:.3f}'.format(ari, v_score))
                word_aris.append(ari)
                word_v_scores.append(v_score)
        if word_aris and word_v_scores:
            print('ARI: {:.3f}, V-score: {:.3f}'.format(
                np.mean(word_aris), np.mean(word_v_scores)))
        aris.extend(word_aris)
        v_scores.extend(word_v_scores)
    if aris and v_scores:
        print()
        print('ARI: {:.3f}, V-score: {:.3f}'.format(
                np.mean(aris), np.mean(v_scores)))


def main():
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--n-senses', type=int, default=6)
    arg('--limit', type=int)
    arg('--n-runs', type=int, default=3)
    arg('--window', type=int, default=10)
    arg('--word')
    params = vars(parser.parse_args())
    print(params)
    run_all(**params)


if __name__ == '__main__':
    main()
