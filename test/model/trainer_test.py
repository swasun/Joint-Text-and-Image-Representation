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
from model.trainer import Trainer
from model.model_builder import ModelBuilder
from representation.text_representation import TextRepresentation
from representation.image_representation import ImageRepresentation

import unittest


class TrainerTest(unittest.TestCase):

    def test_initial_model(self):
        path_captions = ProjectPaths.text_representation_path
        path_embedding_captions = ProjectPaths.embedding_captions_path
        path_embedding_images = ProjectPaths.embedding_images_path
        text_representation = TextRepresentation(path_captions)

        features = ImageRepresentation.load_features(path_embedding_images)

        model_builder = ModelBuilder(text_representation, path_embedding_captions)

        trainer = Trainer(text_representation, features, model_builder, loss='initial', mini_batch_size=50)

        trainer.train(epochs=40)

        trainer.save(
            image_name=ProjectPaths.image_model_init_path,
            caption_name=ProjectPaths.caption_model_init_path,
            caption_representations_name=ProjectPaths.caption_representations_name,
            image_representations_name=ProjectPaths.image_representations_name
        )

    def test_sh_model(self):
        path_captions = ProjectPaths.text_representation_path
        path_embedding_captions = ProjectPaths.embedding_captions_path
        path_embedding_images = ProjectPaths.embedding_images_path
        
        text_representation = TextRepresentation(path_captions)
        features = ImageRepresentation.load_features(path_embedding_images)

        model_builder = ModelBuilder(text_representation, path_embedding_captions)

        trainer = Trainer(text_representation, features, model_builder, loss='sh', mini_batch_size=50)

        trainer.train(epochs=40)

        trainer.save(
            image_name=ProjectPaths.image_model_sh_path,
            caption_name=ProjectPaths.caption_model_sh_path,
            caption_representations_name=ProjectPaths.caption_representations_name,
            image_representations_name=ProjectPaths.image_representations_name
        )

    def test_mh_model(self):
        path_captions = ProjectPaths.text_representation_path
        path_embedding_captions = ProjectPaths.embedding_captions_path
        path_embedding_images = ProjectPaths.embedding_images_path
        text_representation = TextRepresentation(path_captions)

        features = ImageRepresentation.load_features(path_embedding_images)

        model_builder = ModelBuilder(text_representation, path_embedding_captions)

        trainer = Trainer(text_representation, features, model_builder, loss='mh', mini_batch_size=50)

        trainer.train(epochs=40)

        trainer.save(
            image_name=ProjectPaths.image_model_mh_path,
            caption_name=ProjectPaths.caption_model_mh_path,
            caption_representations_name=ProjectPaths.caption_representations_name,
            image_representations_name=ProjectPaths.image_representations_name
        )


if __name__ == '__main__':
    unittest.main()
