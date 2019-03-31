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

import sys
import os
sys.path.append('..' + os.sep + '..' + os.sep + 'src')

from project_paths import ProjectPaths
from tsne.tsne_image_caption_2d_scatter import TSNEImageCaption2DScatter
from representation.text_representation import TextRepresentation

import unittest
import numpy as np
from keras.models import load_model
from keras.applications.resnet50 import ResNet50


class TSNEImageCaption2DScatterTest(unittest.TestCase):

    def test_generate(self):
        tsne_image_caption_scatter = TSNEImageCaption2DScatter(
            input_directory=ProjectPaths.datasets_path + os.sep + '05-caption-images',
            image_activations=np.load(ProjectPaths.image_representations_name + '.npy'),
            caption_activations=np.load(ProjectPaths.caption_representations_name + '.npy'),
            image_model=load_model(ProjectPaths.image_model_path),
            feature_model=ResNet50(weights='imagenet', include_top=False, pooling='avg'),
            text_representation=TextRepresentation(ProjectPaths.text_representation_path)
        )

        tsne_image_caption_scatter.generate()

if __name__ == '__main__':
    unittest.main()
