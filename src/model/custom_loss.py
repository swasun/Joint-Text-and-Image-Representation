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

from keras import backend as K


def vse_loss(y_true, y_pred):
    """
    We didn't use y_true since it's the identity matrix
    """
    positive = y_pred[:, 0]
    negative = y_pred[:, 1]

    return K.sum(K.maximum(0., 1. - positive + negative))

def vse_accuracy(y_true, y_pred):
    """
    We didn't use y_true since it's the identity matrix
    """

    positive = y_pred[:, 0]
    negative = y_pred[:, 1]
    return K.mean(positive > negative)


def sh_loss(alpha, n):
    def _sh_loss(y_true, y_pred):
        positive = K.reshape(K.sum(K.eye(n) * y_pred, axis=0), (n, 1))
        negative_captions = y_pred - K.eye(n) * alpha
        negative_images = K.transpose(negative_captions)

        return K.sum(K.sum(K.maximum(0., alpha - positive + negative_captions), axis=1)
                     + K.sum(K.maximum(0., alpha - positive + negative_images), axis=1))
    return _sh_loss


def mh_loss(alpha, n):
    def _mh_loss(y_true, y_pred):
        positive = K.reshape(K.sum(K.eye(n) * y_pred, axis=0), (n, 1))
        negative_captions = y_pred - K.eye(n) * alpha
        negative_images = K.transpose(negative_captions)

        return K.sum(K.max(K.maximum(0., alpha - positive + negative_captions), axis=1)
                     + K.max(K.maximum(0., alpha - positive + negative_images), axis=1))
    return _mh_loss


def vse_plus_accuracy(n):
    def accuracy(y_true, y_pred):
        positive = K.reshape(K.sum(K.eye(n) * y_pred, axis=0), (n, 1))
        negative = y_pred
        return K.sum(K.cast(positive > negative, dtype=float) * (1-K.eye(n)))/(n*(n-1))
    return accuracy
