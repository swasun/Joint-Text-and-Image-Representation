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
from keras.layers import Layer
from tensorflow.keras.backend import l2_normalize


class CosSimLayer(Layer):
    """
    Layer outputing the cosine similarity between image and caption representation,
    given the input [emb_images, emb_captions]
    """
    
    def __init__(self, **kwargs):
        super(CosSimLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(CosSimLayer, self).build(input_shape)
        
    def call(self, x, **kwargs):
        assert isinstance(x, list)
        images, captions = x
        return [K.dot(images, K.transpose(captions))]
        
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_images, shape_captions = input_shape

        return shape_images[0], shape_captions[0]
