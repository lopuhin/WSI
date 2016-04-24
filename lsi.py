#!/usr/bin/env python
import logging
import argparse

from gensim.models.lsimodel import LsiModel

import utils


def word_lsi(word, num_topics, window, limit=None):
    dictionary, corpus, _ = \
        utils.prepare_corpus(word, window=window, limit=limit)
    lsi = LsiModel(corpus, id2word=dictionary, num_topics=num_topics)
    return lsi


def run_all(*, word, n_runs, limit, n_senses, window):
    words = [word] if word else utils.all_words
    results_by_word = utils.apply_to_words(
        word_lsi, words, n_runs,
        n_senses=n_senses, limit=limit, window=window)
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
