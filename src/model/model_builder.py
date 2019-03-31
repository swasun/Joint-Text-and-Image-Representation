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

from model.embedding import load
from model.custom_loss import vse_loss, vse_accuracy, sh_loss, mh_loss, vse_plus_accuracy
from model.cos_sim_layer import CosSimLayer
from representation.text_representation import TextRepresentation

from keras.layers import Input, Dense, Embedding, GRU, concatenate, dot, Dot
from keras.models import Model


class ModelBuilder(object):

    def __init__(self, text_representation, embedding_path):
        self.embedding_path = embedding_path
        self._text_representation = text_representation

    def build(self, batch_size=50, loss='initial', alpha=0.2):
        if loss is 'initial':
            return self._build_initial()
        else:
            return self._build_vse_plus(loss, batch_size, alpha)

    def _build_initial(self):
        embedding_weights = load(self._text_representation.vocab, 100, self.embedding_path)

        image_input = Input(shape=(2048,))
        caption_input = Input(shape=(16,))
        noise_input = Input(shape=(16,))

        caption_embedding = Embedding(len(self._text_representation.vocab), 100, input_length=16, weights=[embedding_weights])
        caption_rnn = GRU(256)
        image_dense = Dense(256, activation='tanh')

        image_pipeline = image_dense(image_input)
        caption_pipeline = caption_rnn(caption_embedding(caption_input))
        noise_pipeline = caption_rnn(caption_embedding(noise_input))

        positive_pair = dot([image_pipeline, caption_pipeline], axes=-1)
        negative_pair = dot([image_pipeline, noise_pipeline], axes=-1)
        output = concatenate([positive_pair, negative_pair])

        training_model = Model(input=[image_input, caption_input, noise_input], output=output)
        # a model for training which outputs the concatenation
        # of the result of the positive and negative pairs

        image_model = Model(input=image_input, output=image_pipeline)
        # a model which compute a novel image representation

        caption_model = Model(input=caption_input, output=caption_pipeline)
        # a model which compute a caption representation

        training_model.compile(
            loss=vse_loss,
            optimizer='adam',
            metrics=[vse_accuracy]
        )

        return training_model, image_model, caption_model

    def _build_vse_plus(self, loss, batch_size, alpha):

        embedding_weights = load(self._text_representation.vocab, 100, self.embedding_path)

        image_input = Input(shape=(2048,))
        caption_input = Input(shape=(16,))

        caption_embedding = Embedding(len(self._text_representation.vocab), 100, input_length=16, weights=[embedding_weights])
        caption_rnn = GRU(256)
        image_dense = Dense(256, activation='tanh')

        image_pipeline = image_dense(image_input)
        caption_pipeline = caption_rnn(caption_embedding(caption_input))

        output = CosSimLayer()([image_pipeline, caption_pipeline])
        #output = dot([image_pipeline, caption_pipeline], normalize=True, axes=0

        training_model = Model(input=[image_input, caption_input], output=output)
        # a model for training which outputs the concatenation
        # of the result of the positive and negative pairs

        image_model = Model(input=image_input, output=image_pipeline)
        # a model which compute a novel image representation

        caption_model = Model(input=caption_input, output=caption_pipeline)
        # a model which compute a caption representation

        if loss is 'sh':
            fct_loss = sh_loss
        elif loss is 'mh':
            fct_loss = mh_loss
        else:
            raise ValueError("Invalid loss arg")

        training_model.compile(
            loss=fct_loss(alpha, batch_size),
            optimizer='adam',
            metrics=[vse_plus_accuracy(batch_size)]
        )

        return training_model, image_model, caption_model
