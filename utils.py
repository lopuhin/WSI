import os.path
import random
import re
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

from gensim import corpora
import pymystem3
from sklearn.metrics.pairwise import cosine_similarity


all_words = [
    # multiple dictionary senses
    'альбом',
    'билет',
    'блок',
    'вешалка',
    'вилка',
    'винт',
    'горшок',
    # single sense in the dictionary
    'вата',
    'бык',
    'байка',
    'баян',
    'бомба',
    # really single sense
    'борщ',
    'воск',
    'бухгалтер',
    ]


MyStem = pymystem3.Mystem()


def load_stopwords():
    with open('stopwords.txt') as f:
        return {line.strip().split()[0] for line in f if line.strip()}


stopwords = load_stopwords()


def load_contexts(root, word, window=None):
    with open(os.path.join(root, '{}.txt'.format(word))) as f:
        contexts = []
        for line in f:
            left, _, right = line.split('\t')
            left, right = [x.strip().split() for x in [left, right]]
            if window:
                left = left[-window:]
                right = right[:window]
            ctx = left + right
            contexts.append(
                [w for w in ctx if w not in stopwords and w != word])
        return contexts


def weights_contexts(word, window):
    weights = load_weights('../corpora/ad-nouns/cdict/', word)
    contexts = load_contexts('../corpora/ad-nouns-contexts-100k', word, window)
    return weights, contexts


word_re = re.compile(r'\w+', re.U)


def normalize(ctx):
    left, _, right = ctx
    text = ' '.join([left, right]).strip()
    text = re.sub(r'\d', '2', text)
    return [w for w in MyStem.lemmatize(' '.join(word_re.findall(text)))
            if w not in stopwords and w.strip()]


def load_weights(root, word):
    with open(os.path.join(root, word + '.txt')) as f:
        return {w: float(weight) for w, weight in (l.split() for l in f)}


def print_senses(sense_words, topn=5):
    for sense_id, words in sorted(sense_words.items()):
        words = list(words)
        words.sort(key=lambda x: x[1], reverse=True)
        print(sense_id, ' '.join(w for w, _ in words[:topn]), sep='\t')


def print_cluster_sim(centers):
    sim_matrix = cosine_similarity(centers, centers)
    print('\t'.join('{}'.format(j) for j, _ in enumerate(sim_matrix)))
    for i, row in enumerate(sim_matrix):
        print('\t'.join(
            ('{:.2f}'.format(x) if i < j else ' ')
            for j, x in enumerate(row)), i, sep='\t')


def merge_clusters(centers, threshold):
    ''' Merge clusters that are closer then given threshold.
    Return mapping: old clusters -> new clusters.
    '''
    sim_matrix = cosine_similarity(centers, centers)
    mapping = {i: i for i, _ in enumerate(centers)}
    id_gen = len(mapping)
    for i, row in enumerate(sim_matrix):
        for j, sim in enumerate(row):
            if i > j and sim >= threshold:
                # merge (i, j)
                new_id = id_gen
                id_gen += 1
                for id_old in [i, j]:
                    old_new = mapping[id_old]
                    for old, new in list(mapping.items()):
                        if new == old_new:
                            mapping[old] = new_id
    remap = {new: i for i, new in enumerate(set(mapping.values()))}
    return {old: remap[new] for old, new in mapping.items()}


def weights_flt(weights, min_weight, ctx):
    return [w for w in ctx if weights.get(w, 0) > min_weight]


def prepare_corpus(word, *, window, min_weight=1.0, limit=None):
    weights, contexts = weights_contexts(word, window)
    _weights_flt = partial(weights_flt, weights, min_weight)
    contexts = [ctx for ctx in map(weights_flt, contexts) if ctx]
    random.shuffle(contexts)
    if limit:
        contexts = contexts[:limit]
    print(len(contexts))
    dictionary = corpora.Dictionary(contexts)
    corpus = [dictionary.doc2bow(ctx) for ctx in contexts]
    return dictionary, corpus, _weights_flt


def apply_to_words(fn, words, n_runs, **kwargs):
    futures = []
    with ProcessPoolExecutor(max_workers=4) as e:
        for word in words:
            futures.extend(
                (word, e.submit(fn, word, **kwargs))
                for _ in range(n_runs))
    results_by_word = defaultdict(list)
    for word, f in futures:
        results_by_word[word].append(f.result())
    return results_by_word
