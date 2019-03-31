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

from representation.image_representation import ImageRepresentation
from representation.text_representation import TextRepresentation

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import json


class CaptionRetreival(object):

    def __init__(self, text_representation, image_model, caption_model, vocab, caption_representations, features):
        self._text_representation = text_representation
        self._image_model = image_model
        self._caption_model = caption_model
        self._vocab = vocab
        self._caption_representations = caption_representations
        self._features = features

    def generate_caption(self, image_filename=None, n=10):
        # generate image representation for new image
        if image_filename == None:
            image_representation = self._image_model.predict(self._features)
        else:
            image_representation = self._image_model.predict(ImageRepresentation(image_filename).extract_features())
        # compute score of all captions in the dataset
        scores = np.dot(self._caption_representations, image_representation.T).flatten()
        # compute indices of n best captions
        indices = np.argpartition(scores, -n)[-n:]
        indices = indices[np.argsort(scores[indices])]

        print('self._text_representation._texts: ', len(self._text_representation._texts))
        print('indices: ', indices.shape)
        print('scores: ', scores.shape)

        # display them
        return [(scores[i], self._text_representation._texts[i]) for i in [int(x) for x in reversed(indices)]]



    @staticmethod
    def load(text_representation_path, image_model_path, caption_model_path, vocab_path,
            caption_representations_path, features_path=None):
        text_representation = TextRepresentation(text_representation_path)
        image_model = load_model(image_model_path)
        caption_model = load_model(caption_model_path)
        vocab = None
        with open(vocab_path) as vocab_file:
            vocab = json.loads(vocab_file.read())
        caption_representations = np.load(caption_representations_path)
        if features_path:
            features = ImageRepresentation.load_features(features_path)
        else:
            features = None
        return CaptionRetreival(text_representation, image_model, caption_model, vocab, caption_representations, features)
