import os.path
import re

import pymystem3


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
