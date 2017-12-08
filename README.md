# TiSeLaC-ECMLPKDD17
TiSeLaC ECML/PKDD 2017 discovery challenge solution

This repo hosts the first-place solution to the discovery challenge on [**Time Series Land cover Classification (_TiSeLaC_)**](https://sites.google.com/site/dinoienco/tiselc), organized in conjunction of [**ECML-PKDD 2017**](http://ecmlpkdd2017.ijs.si/). 

The challenge consists in predicting the Land Cover class of a set of pixels given their image time series data acquired by the satellites. We propose an *end-to-end* learning approach leveraging *both temporal and spatial information* and requiring very little data preprocessing and feature engineering. 

## architecture
The architecture---ranked first out of 21 teams---comprises different modules using dense multi-layer perceptrons, one-dimensional dilated convolutional and fully connected one-dimensional convolutiona neural layers. 

## requirements
To run, the following libraries are required:

  * [numpy (1.10+)](http://www.numpy.org/),
  * [sklearn (0.18+)](http://scikit-learn.org/stable/),
  * [keras (2.0+)](http://http://keras.io/) (we used [Theano](http://deeplearning.net/software/theano/) as a backend).

## usage
To train on the full training data, and predict for the whole test, you can run:

    ipython -- deep-tsc.py

## paper
The code reproduces the obtained results (collected in `baML.txt`) as reported in the following paper:

_Nicola Di Mauro, Antonio Vergari, Teresa M.A. Basile, Fabrizio G. Ventola, Floriana Esposito_  
[**End-to-end Learning of Deep Spatio-temporal
Representations for Satellite Image Time Series Classification**](http://www.di.uniba.it/~ndm/pubs/dimauro17ecmldc.pdf),  
In:  Proceedings of the ECML/PKDD Discovery Challenges, 2017
