import os.path


def load_stopwords():
    with open('stopwords.txt') as f:
        return {line.strip().split()[0] for line in f if line.strip()}


stopwords = load_stopwords()


def load_contexts(root, word):
    with open(os.path.join(root, '{}.txt'.format(word))) as f:
        return [[w for w in line.strip().split()
                 if w not in stopwords and w != word] for line in f]
