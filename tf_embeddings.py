import argparse
from pathlib import Path

import adagram
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('model')
    arg('output')
    arg('--start', type=int, default=0)
    arg('--stop', type=int, default=1000)
    args = parser.parse_args()

    model = adagram.VectorModel.load(args.model)
    save_embeddings(model, Path(args.output),
                    start=args.start, stop=args.stop)


def save_embeddings(model: adagram.VectorModel, output: Path,
                    start: int, stop: int):
    labels = []
    senses = []
    for word in model.dictionary.id2word[start:stop]:
        for sense, _ in model.word_sense_probs(word):
            labels.append('{} #{}'.format(word, sense))
            senses.append(model.sense_vector(word, sense))
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