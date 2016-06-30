#!/bin/sh

#################
## Loading Data ##
#################
python load_data.py

#################
## Training ##
#################
python train_convert_keras.py

#################
## Predictions ##
#################
python predict_convnet_keras.py

#################
## Submission ##
#################
python create_submission.py
