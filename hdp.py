#!/usr/bin/env python
import logging
import argparse

from gensim.models import HdpModel

import utils


def word_hdp(word, *, window, limit=None):
    dictionary, corpus, _ = \
        utils.prepare_corpus(word, window=window, limit=limit)
    hdp = HdpModel(corpus, id2word=dictionary)
    return hdp


def run_all(*, word, n_runs, limit, n_senses, window):
    words = [word] if word else utils.all_words
    results_by_word = utils.apply_to_words(
        word_hdp, words, n_runs, limit=limit, window=window)
    for word, results in sorted(results_by_word.items()):
        print()
        print(word)
        for hdp in results:
            sense_words = dict(hdp.show_topics(
                topics=n_senses, topn=5, formatted=False))
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
