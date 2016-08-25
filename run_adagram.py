#!/usr/bin/env python
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    with open(args.filename) as f:
        data = json.load(f)
    for word, senses in sorted(data.items()):
        print()
        print(word)
        for idx, sense in enumerate(
                sorted(senses, key=lambda s: s['weight'], reverse=True)):
            print(idx, '{:.2f}'.format(sense['weight']),
                  ' '.join(w for w, _, _ in sense['neighbours']), sep='\t')


if __name__ == '__main__':
    main()
