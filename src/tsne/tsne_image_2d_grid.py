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
import os
from lapjv import lapjv
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from tensorflow.python.keras.preprocessing import image


class TSNEImage2DGrid(object):
    """
    Build the embedded t-SNE space of an image representation,
    and output the representation in a 2D grid image.

    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    https://github.com/prabodhhere/tsne-grid
    https://stackoverflow.com/a/22570069
    """
    
    def __init__(self, input_directory, activations, output_dimension=30,
        out_resolution=224, output_name='tsne_image_2d_grid.jpg',
        output_directory='./', perplexity=50, iterations=5000,
        quality=100):
        
        """
        Parameters
        ----------
        input_directory : str
            Source directory for images

        activations : numpy.ndarray
            Activations of a trained image model

        out_resolution : int, optional (default: 224)
            Width/height of output square image

        output_dimension : int (default: 30)
            Number of small images in a row/column in output image

        output_name : str, optional (default: tsne_image_2d_grid.jpg)
            Name of output image file

        output_directory : str, optional (default: ./)
            Destination directory for output image

        perplexity : int, optional (default: 50)
            t-SNE perplexity

        iterations : int, optional (default: 5000)
            Number of iterations in tsne algorithm

        quality : int, optional (default: 100)
            Quality of the output image
        """
        
        self.output_dimension = output_dimension
        self.input_directory = input_directory
        self.activations = activations
        self.out_resolution = out_resolution
        self.output_name = output_name
        self.output_directory = output_directory
        self.perplexity = perplexity
        self.iterations = iterations
        self.quality = quality
        self.to_plot = np.square(self.output_dimension)

        if self.output_dimension == 1:
            raise ValueError("Output grid dimension 1x1 not supported.")

        if not os.path.exists(self.input_directory):
            raise ValueError("'{}' not a valid directory.".format(self.input_directory))

        if not os.path.exists(self.output_directory):
            raise ValueError("'{}' not a valid directory.".format(self.output_directory))

    def generate(self):
        img_collection = self._load_img(self.input_directory)
        X_2d = self._generate_tsne()
        self._save_tsne_grid(img_collection, X_2d, self.out_resolution, self.output_dimension)

    def _load_img(self, input_directory):
        pred_img = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
        img_collection = []
        for _, img in enumerate(pred_img):
            img = os.path.join(input_directory, img)
            img_collection.append(image.load_img(img, target_size=(self.out_resolution, self.out_resolution)))
        if (np.square(self.output_dimension) > len(img_collection)):
            raise ValueError("Cannot fit {} images in {}x{} grid".format(len(img_collection), self.output_dimension, self.output_dimension))
        return img_collection

    def _generate_tsne(self):
        tsne = TSNE(perplexity=self.perplexity, n_components=2, init='pca', n_iter=self.iterations)
        X_2d = tsne.fit_transform(np.array(self.activations)[0:self.to_plot,:])
        X_2d -= X_2d.min(axis=0)
        X_2d /= X_2d.max(axis=0)
        return X_2d

    def _save_tsne_grid(self, img_collection, X_2d, out_resolution, output_dimension):
        grid = np.dstack(np.meshgrid(np.linspace(0, 1, output_dimension), np.linspace(0, 1, output_dimension))).reshape(-1, 2)
        cost_matrix = cdist(grid, X_2d, "sqeuclidean").astype(np.float32)
        cost_matrix = cost_matrix * (100000 / cost_matrix.max())
        _, col_asses, _ = lapjv(cost_matrix)
        grid_jv = grid[col_asses]
        out = np.ones((output_dimension*out_resolution, output_dimension*out_resolution, 3))

        for pos, img in zip(grid_jv, img_collection[0:self.to_plot]):
            h_range = int(np.floor(pos[0]* (output_dimension - 1) * out_resolution))
            w_range = int(np.floor(pos[1]* (output_dimension - 1) * out_resolution))
            out[h_range:h_range + out_resolution, w_range:w_range + out_resolution] = image.img_to_array(img)

        im = image.array_to_img(out)
        im.save(self.output_directory + self.output_name, quality=self.quality)
