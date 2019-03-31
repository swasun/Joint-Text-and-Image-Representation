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

from representation.imagenet_utils import preprocess_input

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
from tensorflow.python.keras.preprocessing import image


class TSNEImageCaption2DScatter(object):
    """
    Build the embedded t-SNE space of an image representation,
    and output the representation in a 2D scatter image. For
    each image, few likely captions (captions_per_image, default: 2)
    are annoted below the image.


    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne
    https://stackoverflow.com/a/22570069
    """
    
    def __init__(self, input_directory, image_activations, caption_activations, image_model,
        feature_model, text_representation, first_n_images=None, captions_per_image=2, output_image_dimension=15,
        output_caption_dimension=10, out_resolution=96,
        output_name='tsne_image_caption_2d_scatter.jpg', output_directory='./',
        perplexity=50, iterations=5000, output_size=(70, 70), quality=100):
        
        """
        Parameters
        ---------
        input_directory : str
            Source directory for images

        image_activations : numpy.ndarray
            Activations of a trained image model

        caption_activations : numpy.ndarray
            Activations of a trained caption model

        image_model : keras.models.Model
            Our trained image model

        feature_model : keras.models.Model
            Feature model used to build image_activations

        text_representation : TextRepresentation
            Caption dataset representation

        first_n_images : int, optional (default: None)
            Specified the number of first n images to use

        captions_per_image : int, optional (default: 2)
            Number of captions under an image

        output_image_dimension : int, optional (default: 15)
            Dimension of output images

        output_caption_dimension : int, optional (default: 10)
            Dimension of output captions
        
        out_resolution : int, optional (default: 96)
            Width/height of output square image

        output_name : str, optional (default: tsne_image_caption_2d_scatter.jpg)
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
        self.image_activations = image_activations
        self.caption_activations = caption_activations
        self.first_n_images = first_n_images
        self.captions_per_image = captions_per_image
        self.image_model = image_model
        self.feature_model = feature_model
        self.text_representation = text_representation
        self.output_image_dimension = output_image_dimension
        self.output_caption_dimension = output_caption_dimension
        self.out_resolution = out_resolution
        self.output_name = output_name
        self.output_directory = output_directory
        self.perplexity = perplexity
        self.iterations = iterations
        self.output_size = output_size
        self.quality = quality

        if self.output_image_dimension == 1:
            raise ValueError("Output scatter dimension 1x1 not supported.")

        if self.output_caption_dimension == 1:
            raise ValueError("Output scatter dimension 1x1 not supported.")

        if not os.path.exists(self.input_directory):
            raise ValueError("'{}' not a valid directory.".format(self.input_directory))

        if not os.path.exists(self.output_directory):
            raise ValueError("'{}' not a valid directory.".format(self.output_directory))

    def generate(self):
        img_collection = self._load_img(self.input_directory)
        x_image, y_image = self._build_image_space()
        x_caption, y_caption = self._build_caption_space()
        self._plot_tsne_scatter(img_collection, x_image[:self.first_n_images], y_image[:self.first_n_images],
            x_caption[:self.first_n_images], y_caption[:self.first_n_images], self.text_representation._texts)

    def _load_img(self, input_directory):
        pred_img = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
        img_collection = []
        for _, img in enumerate(pred_img):
            img = os.path.join(input_directory, img)
            img_collection.append(image.load_img(img, target_size=(self.out_resolution, self.out_resolution)))
        if (np.square(self.output_image_dimension) > len(img_collection)):
            raise ValueError("Ouput dimension cannot be greater than image number")
        return img_collection

    def _build_image_space(self):
        to_plot = np.square(self.output_image_dimension)

        tsne = TSNE(perplexity=self.perplexity, n_components=2, init='pca', n_iter=self.iterations)
        X_2d = tsne.fit_transform(np.array(self.image_activations)[0:to_plot,:])
        X_2d -= X_2d.min(axis=0)
        X_2d /= X_2d.max(axis=0)

        x = []
        y = []
        for value in X_2d:
            x.append(value[0])
            y.append(value[1])

        return x, y

    def _build_caption_space(self):
        to_plot = np.square(self.output_caption_dimension)

        tsne = TSNE(perplexity=self.perplexity, n_components=2, init='pca', n_iter=self.iterations)
        X_2d = tsne.fit_transform(np.array(self.caption_activations)[0:to_plot,:])
        X_2d -= X_2d.min(axis=0)
        X_2d /= X_2d.max(axis=0)

        x = []
        y = []
        for value in X_2d:
            x.append(value[0])
            y.append(value[1])

        return x, y

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

    def _extract_images_features(self, feature_model, img):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = feature_model.predict(x)
        return np.expand_dims(features.flatten(), axis=0)

    def _plot_tsne_scatter(self, img_collection, x_image, y_image,
        x_caption, y_caption, texts):

        fig = plt.figure(figsize=self.output_size)

        ax = fig.add_subplot(111)

        for i in range(len(x_image)):
            self._imscatter(x_image[i], y_image[i], img_collection[i], ax)
            image_features = self._extract_images_features(self.feature_model, img_collection[i])
            image_representation = self.image_model.predict(image_features)
            scores = np.dot(self.caption_activations, image_representation.T).flatten()
            indices = np.argpartition(scores, -self.captions_per_image)[-self.captions_per_image:]
            indices = indices[np.argsort(scores[indices])]
            annotations = '\n'.join([texts[j] for j in [int(x) for x in reversed(indices)]])
            ax.annotate(annotations,
                xy=(x_image[i], y_image[i]),
                xytext=(5, 2),
                textcoords='offset points',
                ha='right',
                va='bottom')

        plt.savefig(self.output_name, quality=self.quality)
