 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe, Guillaume Ollier, Balthazar Casalé             #
 #                                                                                   #
 # This file is part of Joint-Text-Image-Representation.                             #
 #                                                                                   #
 #   Permission is hereby granted, free of charge, to any person obtaining a copy    #
 #   of this software and associated documentation files (the "Software"), to deal   #
 #   in the Software without restriction, including without limitation the rights    #
 #   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
 #   copies of the Software, and to permit persons to whom the Software is           #
 #   furnished to do so, subject to the following conditions:                        #
 #                                                                                   #
 #   The above copyright notice and this permission notice shall be included in all  #
 #   copies or substantial portions of the Software.                                 #
 #                                                                                   #
 #   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
 #   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
 #   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
 #   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
 #   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
 #   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

from __future__ import print_function
import sys
import numpy as np


def load(vocab, dimension, filename):
    print('loading embeddings from "%s"' % filename, file=sys.stderr)
    embedding = np.zeros((max(vocab.values()) + 1, dimension), dtype=np.float32)
    seen = set()
    with open(filename) as fp:
        for line in fp:
            tokens = line.strip().split(' ')
            if len(tokens) == dimension + 1:
                word = tokens[0]
                if word in vocab:
                    embedding[vocab[word]] = [float(x) for x in tokens[1:]]
                    seen.add(word)
                    if len(seen) == len(vocab):
                        break
    return embedding


if __name__ == '__main__':
    # can be used to filter an embedding file
    if len(sys.argv) != 3:
        print('usage: cat wordlist | %s <dimension> <embedding_filename>' % sys.argv[0])
        sys.exit(1)

    vocab = {word.strip(): i for i, word in enumerate(sys.stdin.readlines())}
    dimension = int(sys.argv[1])
    filename = sys.argv[2]
    embedding = load(vocab, dimension, filename)

    for word, i in vocab.items():
        print(word, ' '.join([str(x) for x in embedding[i]]))
