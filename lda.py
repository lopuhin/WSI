#!/usr/bin/env python
import logging
import argparse
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from gensim import corpora
# from gensim.models.ldamulticore import LdaMulticore as LdaModel
# from gensim.models import HdpModel
from gensim.models import LdaModel
import rl_wsd_labeled
from sklearn.metrics import v_measure_score, adjusted_rand_score

import utils


def word_lda(word, num_topics, limit=None):
    weights = utils.load_weights('../corpora/ad-nouns/cdict/', word)
    texts = utils.load_contexts('../corpora/ad-nouns-contexts-100k', word)
    weights_flt = lambda ctx: [w for w in ctx if weights.get(w, 0) > 1]
    texts = list(map(weights_flt, texts))
    texts = [ctx for ctx in texts if ctx]
    random.shuffle(texts)
    if limit:
        texts = texts[:limit]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    #lda = HdpModel(corpus, id2word=dictionary)
    lda = LdaModel(
        corpus, id2word=dictionary, num_topics=num_topics,
        passes=4, iterations=100, alpha='auto')

    _senses, contexts = rl_wsd_labeled.get_contexts(
        rl_wsd_labeled.contexts_filename('nouns', 'RuTenTen', word))

    documents = [dictionary.doc2bow(weights_flt(utils.normalize(ctx)))
                 for ctx, _ in contexts]
    gamma, _ = lda.inference(documents)
    pred_topics = gamma.argmax(axis=1)
    true_labels = np.array([int(ans) for _, ans in contexts])

    ari = adjusted_rand_score(true_labels, pred_topics)
    v_score = v_measure_score(true_labels, pred_topics)
    return lda, dictionary, ari, v_score


def print_topics(lda, dictionary, topn=5):
    for topic_id in range(lda.num_topics):
        terms = lda.get_topic_terms(topic_id, topn=topn)
        print(topic_id, ' '.join(dictionary[wid] for wid, _ in terms))


def run_all(*, word, n_runs, limit, n_senses):
    words = [word] if word else utils.all_words
    futures = []
    with ProcessPoolExecutor(max_workers=4) as e:
        for word in words:
            futures.extend(
                (word, e.submit(word_lda, word, n_senses, limit=limit))
                for _ in range(n_runs))
    results_by_word = defaultdict(list)
    for word, f in futures:
        results_by_word[word].append(f.result())
    aris, v_scores = [], []
    for word, results in results_by_word.items():
        print()
        print(word)
        word_aris, word_v_scores = [], []
        for lda, dictionary, ari, v_score in results:
            print_topics(lda, dictionary)
            print('ARI: {:.3f}, V-score: {:.3f}'.format(ari, v_score))
            word_aris.append(ari)
            word_v_scores.append(v_score)
        print('ARI: {:.3f}, V-score: {:.3f}'.format(
              np.mean(word_aris), np.mean(word_v_scores)))
        aris.extend(word_aris)
        v_scores.extend(word_v_scores)
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
    arg('--word')
    params = vars(parser.parse_args())
    print(params)
    run_all(**params)


if __name__ == '__main__':
    main()
