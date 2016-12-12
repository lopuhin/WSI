import argparse
from pathlib import Path
from typing import List

import adagram
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('model')
    arg('output')
    arg('--anchor-words', help='File with anchor words')
    arg('--anchor-add', type=int, default=500,
        help='Number of (closest) words to add to anchor words')
    arg('--freq-start', type=int, default=0)
    arg('--freq-stop', type=int, default=1000)
    args = parser.parse_args()

    model = adagram.VectorModel.load(args.model)
    if args.anchor_words:
        words = get_anchor_words(model, args.anchor_words, args.anchor_add)
    else:
        words = get_freq_words(model, args.freq_start, args.freq_stop)
    save_embeddings(model, Path(args.output), words)


def get_freq_words(model, freq_start, freq_stop):
    return model.dictionary.id2word[freq_start:freq_stop]


def get_anchor_words(
        model: adagram.VectorModel,
        anchor_words_file: str,
        n_add: int,
    ) -> List[str]:
    """ Return anchor words and words closest to them (from n_add closest senses).
    """
    with open(anchor_words_file, 'rt') as f:
        anchor_words = [line.strip() for line in f]
    n_senses = model.In.shape[1]
    dim = model.In.shape[-1]
    anchor_indices = []
    for w in anchor_words:
        idx = model.dictionary.word2id.get(w)
        if idx is not None:
            for sense_idx, _ in model.word_sense_probs(w):
                anchor_indices.append(idx * n_senses + sense_idx)
    senses = model.In.reshape(-1, dim) / model.InNorms.reshape(-1)[:, None]
    anchor_senses = senses[anchor_indices]
    closenesses = senses @ anchor_senses.T
    closenesses[np.isnan(closenesses)] = 0
    closenesses = closenesses.max(axis=1)
    add_indices = np.argpartition(closenesses, -n_add)[-n_add:]
    return list({model.dictionary.id2word[idx // n_senses]
                 for idx in add_indices})


def save_embeddings(model: adagram.VectorModel, output: Path, words: List[str]):
    labels = []
    senses = []
    for word in words:
        for sense, _ in model.word_sense_probs(word):
            labels.append('{} #{}'.format(word, sense))
            v = model.sense_vector(word, sense)
            senses.append(v / np.linalg.norm(v))
    output.mkdir(exist_ok=True)
    labels_path = output.joinpath('labels.tsv')
    labels_path.write_text('\n'.join(labels))
    senses = np.array(senses)

    with tf.Session() as session:
        embedding_var = tf.Variable(senses, trainable=False, name='senses')
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(session, str(output.joinpath('model.ckpt')))

        summary_writer = tf.train.SummaryWriter(str(output))
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = str(labels_path)
        projector.visualize_embeddings(summary_writer, config)


if __name__ == '__main__':
    main()