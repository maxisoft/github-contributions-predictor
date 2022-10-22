Github Activity Predictor
-----------------
A toy project to see how predictable I'm in my so called GitHub contributions ;)


## Technical Process Overview
1. Gather contributions data
2. Train a machine learning model
3. Use the model to predict futures contributions ([published here](https://github.com/maxisoft/github-activity-predictor/tree/pred_history_no_scaling))
4. Repeat `3.` every day by using GitHub actions

## Requirements to rebuild a model
- [anaconda](https://www.anaconda.com/products/distribution)
- [pytorch](https://pytorch.org/get-started/locally/) (with or without GPU)
- any additional pip requirements are listed in `requirements.txt`


## Source files description
To allow one to build his own model the project is organized in multiple ordered python/jupyter files designed to be ran sequentially.

### 0-gather_data.py
Download and save users' contributions and other stats provided by github public api.  
User list is collected by randomly walking the users' *following*/*followers* graph.  
Produce a big `contribs.json` files containing raw users data.  
This script can be ran again to gather even more data.

### 1-pack-data.py
Parse and pack gathered data into numpy ndarrays.  
Produce a compressed `userdata.npz` numpy file

### 2-preprocess.py
Pre-process users' contributions by using the following scheme:  
- data augmentations using `mean`, `std`, `skewness` and `fft`
- outliers removal using quantiles filters mainly
- features normalization using [scikit-learn preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html) tools

Produce a compressed `ml.npz` numpy file and a `scalers.pkl.z` containing pickled scalers.

### 3-train-model.ipynb
Jupyter notebook (designed to be ran on kaggle) for training a pytorch model.

### 4-inference.py
Use previous pytorch model, download latest users' data and predict their contributions number for the next 7 days.  
Produce `csv` files containing predictions.