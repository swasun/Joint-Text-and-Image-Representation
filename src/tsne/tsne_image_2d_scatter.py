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
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
from tensorflow.python.keras.preprocessing import image


class TSNEImage2DScatter(object):
    """
    Build the embedded t-SNE space of an image representation,
    and output the representation in a 2D scatter image.

    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    https://stackoverflow.com/a/22570069
    """
    
    def __init__(self, input_directory, activations, output_dimension=15,
        out_resolution=96, output_name='tsne_image_2d_scatter.jpg',
        output_directory='./', perplexity=50, iterations=5000,
        output_size=(70, 70), quality=100):
        
        """
        Parameters
        ---------
        input_directory : str
            Source directory for images

        activations : numpy.ndarray
            Activations of a trained image model

        output_dimension : int (default: 15)
            Number of small images in output image
        
        out_resolution : int, optional (default: 96)
            Width/height of output square image

        output_name : str, optional (default: tsne_image_2d_scatter.jpg)
            Name of output image file

        output_directory : str, optional (default: ./)
            Destination directory for output image

        perplexity : int, optional (default: 50)
            t-SNE perplexity

        iterations : int, optional (default: 5000)
            Number of iterations in tsne algorithm

        output_size : (int, int), optional (default: (70, 70))
            The size (width, height) of the output image

        quality : int, optional (default: 100)
            Quality of the output image
        """
        
        self.input_directory = input_directory
        self.activations = activations
        self.output_dimension = output_dimension
        self.out_resolution = out_resolution
        self.output_name = output_name
        self.output_directory = output_directory
        self.perplexity = perplexity
        self.iterations = iterations
        self.output_size = output_size
        self.quality = quality
        self.to_plot = np.square(self.output_dimension)

        if self.output_dimension == 1:
            raise ValueError("Output scatter dimension 1x1 not supported.")

        if not os.path.exists(self.input_directory):
            raise ValueError("'{}' not a valid directory.".format(self.input_directory))

        if not os.path.exists(self.output_directory):
            raise ValueError("'{}' not a valid directory.".format(self.output_directory))

    def generate(self):
        img_collection = self._load_img(self.input_directory)
        X_2d = self._generate_tsne()
        self._plot_tsne_scatter(img_collection, X_2d, self.out_resolution, self.output_dimension)

    def _load_img(self, input_directory):
        pred_img = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
        img_collection = []
        for _, img in enumerate(pred_img):
            img = os.path.join(input_directory, img)
            img_collection.append(image.load_img(img, target_size=(self.out_resolution, self.out_resolution)))
        if (np.square(self.output_dimension) > len(img_collection)):
            raise ValueError("Ouput dimension cannot be greater than image number")
        return img_collection

    def _generate_tsne(self):
        tsne = TSNE(perplexity=self.perplexity, n_components=2, init='pca', n_iter=self.iterations)
        X_2d = tsne.fit_transform(np.array(self.activations)[0:self.to_plot,:])
        X_2d -= X_2d.min(axis=0)
        X_2d /= X_2d.max(axis=0)
        return X_2d

    def _imscatter(self, x, y, image, ax=None, zoom=1):
        if ax is None:
            ax = plt.gca()
        try:
            image = plt.imread(image)
        except TypeError:
            pass
        im = OffsetImage(image, zoom=zoom)
        x, y = np.atleast_1d(x, y)
        artists = []
        for x0, y0 in zip(x, y):
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()
        return artists

    def _plot_tsne_scatter(self, img_collection, X_2d, out_resolution, output_dimension):
        x = []
        y = []
        for value in X_2d:
            x.append(value[0])
            y.append(value[1])

        fig = plt.figure(figsize=self.output_size)

        ax = fig.add_subplot(111)

        for i in range(len(x)):
            self._imscatter(x[i], y[i], img_collection[i], ax)

        plt.savefig(self.output_name, quality=self.quality)
