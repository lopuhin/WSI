#!/usr/bin/env python
import logging
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


def word_lda(word, num_topics=6, limit=None):
    weights = utils.load_weights('../corpora/ad-nouns/cdict/', word)
    texts = utils.load_contexts('../corpora/ad-nouns-contexts-100k', word)
    texts = [[w for w in ctx if weights.get(w, 0) > 1] for ctx in texts]
    texts = [ctx for ctx in texts if ctx]
    if limit:
        texts = texts[:limit]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    #lda = HdpModel(corpus, id2word=dictionary)
    lda = LdaModel(
        corpus, id2word=dictionary, num_topics=num_topics,
        passes=3, iterations=100, alpha='auto')

    _senses, contexts = rl_wsd_labeled.get_contexts(
        rl_wsd_labeled.contexts_filename('nouns', 'RuTenTen', word))

    documents = [dictionary.doc2bow(utils.normalize(ctx)) for ctx, _ in contexts]
    gamma, _ = lda.inference(documents)
    pred_topics = gamma.argmax(axis=1)
    true_labels = np.array([int(ans) for _, ans in contexts])

    ari = adjusted_rand_score(true_labels, pred_topics)
    v_score = v_measure_score(true_labels, pred_topics)
    return lda, dictionary, ari, v_score


def print_topics(lda, dictionary):
    for topic_id in range(lda.num_topics):
        terms = lda.get_topic_terms(topic_id, topn=5)
        print(topic_id, ' '.join(dictionary[wid] for wid, _ in terms))


def run_all(n_runs=3, limit=None):
    print('limit', limit)
    futures = []
    with ProcessPoolExecutor(max_workers=4) as e:
        for word in words_1:
            futures.extend((word, e.submit(word_lda, word, limit=limit))
                           for _ in range(n_runs))
    results_by_word = defaultdict(list)
    for word, f in futures:
        results_by_word[word].append(f.result())
    for word, results in results_by_word.items():
        print()
        print(word)
        for lda, dictionary, ari, v_score in results:
            print_topics(lda, dictionary)
            print('ARI: {:.3f}, V-score: {:.3f}'.format(ari, v_score))


words_1 = [
    'альбом',
    'билет',
    'блок',
    'вешалка',
    'вилка',
    'винт',
    'горшок',
    ]


def main():
   #logging.basicConfig(level=logging.INFO)
    run_all(limit=30000)


if __name__ == '__main__':
    main()
