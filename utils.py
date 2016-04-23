import os.path
import re

from sklearn.metrics.pairwise import cosine_similarity
import pymystem3


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


def load_contexts(root, word):
    with open(os.path.join(root, '{}.txt'.format(word))) as f:
        return [[w for w in line.strip().split()
                 if w not in stopwords and w != word] for line in f]


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


def print_cluster_sim(centers):
    sim_matrix = cosine_similarity(centers, centers)
    print(' '.join('{}   '.format(j) for j, _ in enumerate(sim_matrix)))
    for i, row in enumerate(sim_matrix):
        print(' '.join(
            ('{:.2f}'.format(x) if i < j else '    ')
            for j, x in enumerate(row)), i)


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
