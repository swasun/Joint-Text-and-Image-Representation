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

from representation.text_representation import TextRepresentation

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import json

class ImageRetreival(object):

    def __init__(self, caption_model, image_representations, text_representation, vocab):
        self._caption_model = caption_model
        self._image_representations = image_representations
        self._text_representation = text_representation
        self._vocab = vocab

    def search_image(self, caption_path, n=10):
        caption_representation = self._caption_model.predict(self._preprocess_texts([caption_path]))
        scores = np.dot(self._image_representations, caption_representation.T).flatten()
        indices = np.argpartition(scores, -n)[-n:]
        indices = indices[np.argsort(scores[indices])]

        return [(scores[i], self._text_representation._images[i]) for i in [int(x) for x in reversed(indices)]]

    @staticmethod
    def load(caption_model_path, image_representation_path, text_representation_path, vocab_path):
        caption_model = load_model(caption_model_path)
        image_representations = np.load(image_representation_path)
        text_representation = TextRepresentation(text_representation_path)
        vocab = None
        with open(vocab_path) as vocab_file:
            vocab = json.loads(vocab_file.read())
        return ImageRetreival(caption_model, image_representations, text_representation, vocab)

    def _preprocess_texts(self, texts):
        output = []
        for text in texts:
            output.append([self._vocab[word] if word in self._vocab else 0 for word in text.split()])
        return pad_sequences(output, maxlen=16)
