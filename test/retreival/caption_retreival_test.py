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
from retreival.caption_retreival import CaptionRetreival

import unittest


class CaptionRetreivalTest(unittest.TestCase):

    def test_loading(self):
        caption_retreival = CaptionRetreival.load(
            text_representation_path=ProjectPaths.text_representation_path,
            image_model_path=ProjectPaths.image_model_path,
            caption_model_path=ProjectPaths.image_model_path,
            vocab_path=ProjectPaths.vocab_path,
            caption_representations_path=ProjectPaths.caption_representations_name + '.npy',
            features_path=None, # TODO: replace by pre computed dataset features
        )

        print(caption_retreival.generate_caption(ProjectPaths.datasets_path + os.sep + '05-caption-images/COCO_val2014_000000004554.jpg'))

if __name__ == '__main__':
    unittest.main()
