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

import numpy as np
import math


class Trainer(object):

    def __init__(self, text_representation, features, model_builder, loss='initial', mini_batch_size=64, patience=3):
        self._text_representation = text_representation
        self._features = features
        self._training_model, self._image_model, self._caption_model = model_builder.build(mini_batch_size, loss=loss)
        self._loss = loss
        self._batch_size = mini_batch_size
        self._patience = patience

    def train(self, epochs):
        """
        Explanation of using noise:
        We want the dot product between the
        image and text representations to be high when the caption and image match,
        and low when they don't. So our model will compute the dot product of the
        representations of two pairs, a matched (or positive) pair and a mismatched (or
        negative) pair. Then, we will be trained with a maximum margin loss which
        makes sure the positive pair has a (dot product) value higher than one plus the
        value of the negative pair.
        """

        fake_labels = np.zeros((len(self._features), 1))
        noise = None
        noise_train = None

        feat_train = self._features[:9000]
        feat_valid = self._features[-1000:]
        text_train = self._text_representation.captions[:9000]
        text_valid = self._text_representation.captions[-1000:]

        if self._loss is 'initial':
            noise = np.copy(self._text_representation.captions)
            noise_train = noise[:9000]
            noise_valid = noise[-1000:]
            np.random.shuffle(noise)

        y_valid = fake_labels[-1000:]

        rounds_without_improvement = 0
        best_loss = 1e6
        minibatches = self._compute_minibatches(feat_train, text_train, noise_train)
        remainder = feat_valid.shape[0] % self._batch_size
        remainder = None if remainder is 0 else -remainder
        precision = 1000.0

        history = dict()
        history['train_loss'] = list()
        history['train_accuracy'] = list()
        history['val_loss'] = list()
        history['val_accuracy'] = list()

        for epoch in range(epochs):
            losses = list()
            accuracies = list()
            val_losses = list()
            val_accuracies = list()

            if self._loss is 'initial':
                for feat, text, noise, y_dummy in minibatches:
                    self._train_on_batch_iteration(
                        [feat, text, noise],
                        y_dummy,
                        [feat_valid[:remainder], text_valid[:remainder], noise_valid[:remainder]],
                        y_valid[:remainder],
                        losses,
                        accuracies,
                        val_losses,
                        val_accuracies
                    )
            else:
                for feat, text, y_dummy in minibatches:
                    self._train_on_batch_iteration(
                        [feat, text],
                        y_dummy,
                        [feat_valid[:remainder], text_valid[:remainder]],
                        y_valid[:remainder],
                        losses,
                        accuracies,
                        val_losses,
                        val_accuracies
                    )
            
            mean_loss = np.mean(losses)
            mean_accuracy = np.mean(accuracies)
            mean_val_loss = np.mean(val_losses)
            mean_val_accuracy = np.mean(val_accuracies)

            history['train_loss'].append(mean_loss)
            history['train_accuracy'].append(mean_accuracy)
            history['val_loss'].append(mean_val_loss)
            history['val_accuracy'].append(mean_val_accuracy)

            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            if rounds_without_improvement == self._patience:
                print('Early stopped at epoch: {} with loss: {} accuracy: {} val_loss: {} val_accuracy: {}'.format(
                    epoch+1, math.floor(mean_loss * precision) / precision, math.floor(mean_accuracy * precision) / precision,
                    math.floor(mean_val_loss * precision) / precision, math.floor(mean_val_accuracy * precision) / precision))
                break

            print('Epoch {}: loss: {} accuracy: {} val_loss: {} val_accuracy: {}'.format(
                epoch+1, math.floor(mean_loss * precision) / precision, math.floor(mean_accuracy * precision) / precision,
                math.floor(mean_val_loss * precision) / precision, math.floor(mean_val_accuracy * precision) / precision))

        return history

    def save(self, image_name='model.image', caption_name='model.caption',
             caption_representations_name='caption-representations',
             image_representations_name='image-representations'):
        self._image_model.save(image_name)
        self._caption_model.save(caption_name)
        np.save(caption_representations_name, self._caption_model.predict(self._text_representation.captions))
        np.save(image_representations_name, self._image_model.predict(self._features))

    def _train_on_batch_iteration(self, X_train, y_train, X_valid, y_valid, losses, accuracies, val_losses, val_accuracies):
        loss, accuracy = self._training_model.train_on_batch(X_train, y_train)
        eval_loss, eval_accuracy = self._training_model.evaluate(
            X_valid,
            y_valid,
            verbose=None,
            batch_size=self._batch_size
        )
        losses.append(loss)
        accuracies.append(accuracy)
        val_losses.append(eval_loss)
        val_accuracies.append(eval_accuracy)

    def _compute_minibatches(self, feat, text, noise=None):
        minibatches = list()

        order = np.arange(feat.shape[0]) # List the indices of the lines of X
        np.random.shuffle(order) # Shuffle the indices
        
        # Create new temporary vectors with shuffled indices
        feat = feat[order]
        text = text[order]
        noise = noise[order] if noise is not None else None

        # Create minibatches with specified size using the shuffled vectors
        for i in range(0, feat.shape[0], self._batch_size):
            feat_i = feat[i:i + self._batch_size]
            text_i = text[i:i + self._batch_size]
            noise_i = noise[i:i + self._batch_size] if noise is not None else None
            y_dummy = np.full(self._batch_size, 0)

            if noise_i is not None:
                minibatches.append((feat_i, text_i, noise_i, y_dummy))
            else:
                minibatches.append((feat_i, text_i, y_dummy))

        if len(minibatches[len(minibatches) - 1]) != self._batch_size:
            minibatches = minibatches[:len(minibatches)-2]

        return minibatches

    @property
    def image_model(self):
        return self._image_model

    @property
    def caption_model(self):
        return self._caption_model
