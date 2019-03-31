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

import os
import sys
sys.path.append('..' + os.sep + '..' + os.sep + 'src')

from project_paths import ProjectPaths
from representation.text_representation import TextRepresentation
from representation.image_representation import ImageRepresentation
from model.trainer import Trainer
from model.model_builder import ModelBuilder
from retreival.caption_retreival import CaptionRetreival

import unittest


class ModelsTest(unittest.TestCase):

    def test_init(self):
        caption_retreival = CaptionRetreival.load(
            text_representation_path=ProjectPaths.text_representation_path,
            image_model_path=ProjectPaths.image_model_init_path,
            caption_model_path=ProjectPaths.image_model_init_path,
            vocab_path=ProjectPaths.vocab_path,
            caption_representations_path=ProjectPaths.caption_representations_name + '.npy'
        )


        for image, caption in X:
            image_representation = self._image_init_model.predict(ImageRepresentation(image_filename).extract_features())




if __name__ == '__main__':
    unittest.main()
