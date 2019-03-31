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

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json
import numpy as np


class TextRepresentation(object):

    def __init__(self, captions_dataset_path):
        self.captions_dataset_path = captions_dataset_path

        self._texts, self._images = self._load_annotations_dataset(captions_dataset_path)

        _tokenizer = Tokenizer()
        _tokenizer.fit_on_texts(self._texts)

        _sequences = _tokenizer.texts_to_sequences(self._texts)
        self._captions = pad_sequences(_sequences, maxlen=16)

        self._vocab = _tokenizer.word_index
        self._vocab['<eos>'] = 0  # add word with id 0

    def write_vocab(self, path):
        # save the vocab
        with open(path, 'w') as fp:
            fp.write(json.dumps(self.vocab))

    def _load_annotations_dataset(self, captions_filename):
        # return text captions which has already been tokenized,
        # lower-cased and stripped of its punctuation (using
        # NLTK)
        texts = []
        images = []

        with open(captions_filename) as fp:
            for line in fp:
                tokens = line.strip().split()
                images.append(tokens[0])
                texts.append(' '.join(tokens[1:]))

        return texts, images

    @property
    def captions(self):
        return self._captions

    @captions.setter
    def captions(self, value):
        raise AttributeError("can not set attribute caption")

    @property
    def vocab(self):
        return self._vocab

    @vocab.setter
    def vocab(self, value):
        raise AttributeError("can not set attribute vocab")

    @property
    def texts(self):
        return self._texts

    @texts.setter
    def texts(self, value):
        raise AttributeError("can not set attribute texts")
