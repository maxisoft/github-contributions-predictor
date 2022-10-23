GitHub's contributions Predictor
-----------------
A toy project to see how predictable I'm with my so-called GitHub contributions ;)

One of the main goal of this repo is to predict current/next day [contributions](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-profile/managing-contribution-settings-on-your-profile/why-are-my-contributions-not-showing-up-on-my-profile) of multiples users in a daily automated way using GitHub actions.  
To do so this project feature a pytorch model trained with contributions data from GitHub users.  
The history of those predictions is [available in the `pred_history_no_scaling` branch](https://github.com/maxisoft/github-contributions-predictor/tree/pred_history_no_scaling)

## How to get predictions for your contributions
Please first **consider** that this project is just for fun, not well tested and intended for an **harmless use**.

To add a user for the next predictions, do the following:
- [fork this repository](https://github.com/maxisoft/github-contributions-predictor/fork)
- append your GitHub nickname into the `users.txt` file
- commit
- open a pull request

## Technical Process Overview
The following readme parts are now more technical.   

Here's an overview of the process to predict contributions from zero:
1. Gather contributions data
2. Train a machine learning model
3. Use the model to predict futures contributions ([published here](https://github.com/maxisoft/github-contributions-predictor/tree/pred_history_no_scaling))
4. Repeat `3.` every day by using GitHub actions

## Requirements to rebuild a model
- [anaconda](https://www.anaconda.com/products/distribution)
- [pytorch](https://pytorch.org/get-started/locally/) (with or without GPU)
- any additional pip requirements are listed in `requirements.txt`


## Source files description
To allow one to build his own model the project is organized in multiple ordered python/jupyter files designed to be ran sequentially.

### 0-gather_data.py
Download and save users' contributions and other stats provided by GitHub public api.  
User list is collected by randomly walking the users' *following*/*followers* graph.  
Produce a big `contribs.json` files containing raw users' data.  
This script can be run again to gather even more data.

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
Jupyter notebook (designed to be run on kaggle) for training a pytorch model.

### 4-inference.py
Use previous pytorch model, download the latest users' data and predict their contributions number for the next 7 days.  
Produce `csv` files containing predictions.


## Bonus for the readers
There's [an additional branch `pred_history_with_scaling`](https://github.com/maxisoft/github-contributions-predictor/tree/pred_history_with_scaling) containing predictions with a model trained to expect more contributions from users.