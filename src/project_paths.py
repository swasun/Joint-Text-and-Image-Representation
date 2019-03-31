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


class ProjectPaths(object):

    datasets_path = '..' + os.sep + '..' + os.sep + 'datasets'
    text_representation_path = datasets_path + os.sep + '05-caption' + os.sep + 'annotations.10k.txt'
    embedding_captions_path = datasets_path + os.sep + '05-caption' + os.sep + 'glove.twitter.27B.100d.filtered.txt'
    embedding_images_path = datasets_path + os.sep + '05-caption' + os.sep + 'resnet50-features.10k.npy'

    image_model_init_path = datasets_path + os.sep + '05-caption' + os.sep + 'model.init.image'
    caption_model_init_path = datasets_path + os.sep + '05-caption' + os.sep + 'model.init.caption'

    image_model_mh_path = datasets_path + os.sep + '05-caption' + os.sep + 'model.mh.image'
    caption_model_mh_path = datasets_path + os.sep + '05-caption' + os.sep + 'model.mh.caption'

    image_model_sh_path = datasets_path + os.sep + '05-caption' + os.sep + 'model.sh.image'
    caption_model_sh_path = datasets_path + os.sep + '05-caption' + os.sep + 'model.sh.caption'


    caption_representations_name = datasets_path + os.sep + '05-caption' + os.sep + 'caption-representations'
    image_representations_name = datasets_path + os.sep + '05-caption' + os.sep + 'image-representations'
    vocab_path = datasets_path + os.sep + '05-caption' + os.sep + 'vocab.json'

    @staticmethod
    def update_datasets_path(new_datasets_path):
        ProjectPaths.datasets_path = new_datasets_path
        ProjectPaths.text_representation_path = ProjectPaths.datasets_path + os.sep + '05-caption' + os.sep + 'annotations.10k.txt'
        ProjectPaths.embedding_captions_path = ProjectPaths.datasets_path + os.sep + '05-caption' + os.sep + 'glove.twitter.27B.100d.filtered.txt'
        ProjectPaths.embedding_images_path = ProjectPaths.datasets_path + os.sep + '05-caption' + os.sep + 'resnet50-features.10k.npy'
        ProjectPaths.image_model_path = ProjectPaths.datasets_path + os.sep + '05-caption' + os.sep + 'model.image'
        ProjectPaths.caption_model_path = ProjectPaths.datasets_path + os.sep + '05-caption' + os.sep + 'model.caption'
        ProjectPaths.caption_representations_name = ProjectPaths.datasets_path + os.sep + '05-caption' + os.sep + 'caption-representations'
        ProjectPaths.image_representations_name = ProjectPaths.datasets_path + os.sep + '05-caption' + os.sep + 'image-representations'
        ProjectPaths.vocab_path = ProjectPaths.datasets_path + os.sep + '05-caption' + os.sep + 'vocab.json'
