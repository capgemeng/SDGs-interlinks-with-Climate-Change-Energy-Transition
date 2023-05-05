'''
NAME
    train

AUTHOR
    Miguel Chamochin

DATE
    5 Mayo 2023

VERSION
    v0 preliminar

DESCRIPTION
    Main program ML

REQUIREMENTS
    See file ./notebooks/project_resume.ipynb

FILE
./train.py

This script trains the chosen model in 'my_model'.
The chosen model is the model already trained in 'Modelo_Pipeline_PCA_DecissionTree_Regression.py' 
and ready to put into production.

The coefficient of determination of the prediction in Test for this model is 0.9484216064860829
'''

import sys
import os

import json

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '\\utils')

from functions import load_model



loaded_model, X_test, y_test = load_model('my_model')

print('The coefficient of determination of the prediction in Test (loaded_model.score) is:', loaded_model.score(X_test, y_test))