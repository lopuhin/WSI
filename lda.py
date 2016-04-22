#!/usr/bin/env python
import logging

import numpy as np
from gensim import corpora
# from gensim.models.ldamulticore import LdaMulticore as LdaModel
# from gensim.models import HdpModel
from gensim.models import LdaModel
import rl_wsd_labeled
from sklearn.metrics import v_measure_score, adjusted_rand_score

import utils


def word_lda(word, num_topics=6):
    weights = utils.load_weights('../corpora/ad-nouns/cdict/', word)
    texts = utils.load_contexts('../corpora/ad-nouns-contexts-100k', word)
    texts = [[w for w in ctx if weights.get(w, 0) > 1] for ctx in texts]
    texts = [ctx for ctx in texts if ctx]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    print(len(corpus))
    #lda = HdpModel(corpus, id2word=dictionary)
    lda = LdaModel(
        corpus, id2word=dictionary, num_topics=num_topics,
        passes=3, iterations=100, alpha='auto')

    for topic_id in range(lda.num_topics):
        terms = lda.get_topic_terms(topic_id, topn=5)
        print(topic_id, ' '.join(dictionary[wid] for wid, _ in terms))

    _senses, contexts = rl_wsd_labeled.get_contexts(
        rl_wsd_labeled.contexts_filename('nouns', 'RuTenTen', word))

    documents = [dictionary.doc2bow(utils.normalize(ctx)) for ctx, _ in contexts]
    gamma, _ = lda.inference(documents)
    pred_topics = gamma.argmax(axis=1)
    true_labels = np.array([int(ans) for _, ans in contexts])

    ari = adjusted_rand_score(true_labels, pred_topics)
    v_score = v_measure_score(true_labels, pred_topics)
    return ari, v_score


def check_stability():
    for word in ['горшок']:
        print()
        print(word)
        for _ in range(3):
            ari, v_score = word_lda(word)
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
    check_stability()

if __name__ == '__main__':
    main()
