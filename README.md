# Overview

This project contains an implementation of VSE++ losses by [Faghri, Fartash et al., 2017] that is a technique for learning visual-semantic embeddings for cross-modal retrieval, and an implementation of t-SNE by [van der Maaten et al., 2008] (school project, Signal Learning and Multimedia class, 2019).

It is applied on MSCOCO image captioning dataset by [Lin, Tsung-Yi et al., 2014], in particular with the val2014 data which contains a set of 40k images annotated with five captions each. We also used Resnet50 features by [He, Kaiming et al., 2016] and glove embeddings by [Pennington, Jeffrey et al., 2014].

A good introduction of Representation Learning would be [Bengio, Y. et al., 2013].

# Features

* [Image retreival](src/retreival/image_retreival.py)
* [Caption retreival](src/retreival/caption_retreival.py)
* [VSE++ losses](src/model/custom_loss.py) (Loss-Sum-Hinge and Loss-Max-Hinge)
* [t-SNE for captions in a 2D scatter](src/tsne/tsne_caption_2d_scatter.py)
* [t-SNE for captions in 3D scatter](src/tsne/tsne_caption_3d_scatter.py)
* [t-SNE for images in a 2D grid](src/tsne/tsne_image_2d_grid.py)
* [t-SNE for images in a 2D scatter](src/tsne/tsne_image_2d_scatter.py)
* [t-SNE for both captions and images in a 2D scatter](src/tsne/tsne_image_caption_2d_scatter.py)

# Installation

It requires python3, python3-pip, the packages listed in [requirements.txt](requirements.txt) and a recent version of git that supports [git-lfs](https://git-lfs.github.com/).

To install the required packages:
```bash
pip3 install -r requirements.txt
```

# Usage

A [notebook](notebook/notebook.ipynb) is available, and each feature is illustrated in an example in [test](test) directory.

# References

* [Faghri, Fartash et al., 2017] [Faghri, Fartash et al. “VSE++: Improving Visual-Semantic Embeddings with Hard Negatives.” BMVC (2017)](https://arxiv.org/abs/1707.05612).

* [van der Maaten et al., 2008] [van der Maaten, L.J.P.; Hinton, G.E. (Nov 2008). "Visualizing Data Using t-SNE" (PDF). Journal of Machine Learning Research. 9: 2579–2605](http://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf).

* [Lin, Tsung-Yi et al., 2014] [Lin, Tsung-Yi et al. “Microsoft COCO: Common Objects in Context.” Lecture Notes in Computer Science (2014): 740–755. Crossref. Web](https://arxiv.org/abs/1405.0312).

* [Bengio, Y. et al., 2013] [Bengio, Y., A. Courville, and P. Vincent. “Representation Learning: A Review and New Perspectives.” IEEE Transactions on Pattern Analysis and Machine Intelligence 35.8 (2013): 1798–1828. Crossref. Web](https://arxiv.org/abs/1206.5538).

* [He, Kaiming et al., 2016] [He, Kaiming et al. “Deep Residual Learning for Image Recognition.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016): n. pag. Crossref. Web](https://arxiv.org/abs/1512.03385).

* [Pennington, Jeffrey et al., 2014] [Pennington, Jeffrey & Socher, Richard & Manning, Christoper. (2014). Glove: Global Vectors for Word Representation. EMNLP. 14. 1532-1543. 10.3115/v1/D14-1162](https://www.researchgate.net/publication/284576917_Glove_Global_Vectors_for_Word_Representation).

# Authors

* Charly Lamothe
* Guillaume Ollier
* Balthazar Casalé
