'''
NAME
    Modelo_Pipeline_PCA_KNN_Cassification

AUTHOR
    Miguel Chamochin

DATE
    5 Mayo 2023

VERSION
    v0 preliminar

DESCRIPTION
    Code for the Pipeline PCA k-Nearest Neighbours Classification model

REQUIREMENTS
    See file ./notebooks/project_resume.ipynb

FILE
    ./Modelo_Pipeline_PCA_KNN_Cassification.py

This script contains the code for the Pipeline PCA k-Nearest Neighbours Classification model.
Train and test data is read and a pipeline is created with the right parameters. 
Coefficient of Determination for train and test are calculated.
The model has been dumped in the ./model directory. Three files have been generated:
    <modelname>: the model dumped by pickle
    <modelname>.json: a JSON with the description (according to the previous delivery).
    <modelname>.csv: contains the test data for the independent and dependent variables.
The model is then restored and recualculated the scoring for checking. 
'''

# Import the required libraries
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys
import os

import json

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '\\utils')

from functions import read_train_test_split_for_regression, read_train_test_split_for_classification, dump_model, load_model
from variables import fileXStrScores, fileXStrRanks, fileYStr

X_train, X_test, y_train, y_test = read_train_test_split_for_classification(fileXStrScores, fileYStr)

pipeline = Pipeline(steps = [
    ("scaler", StandardScaler()), # primero escalo
    ("pca", PCA()), # segundo aplica PCA 
    ("kneighborsclassifier", KNeighborsClassifier()) # Despues un KNeighborsClassifier
])

pipeline_param = {
    'pca__n_components' :  [1],
    'pca__random_state' :  [42],
    'kneighborsclassifier__n_neighbors' :  [3]
}

gs_pipeline = GridSearchCV(pipeline,
                            pipeline_param,
                            cv = 10,
                            scoring = 'accuracy',
                            verbose = 1, # mensajes del output
                            n_jobs = -1)

search = GridSearchCV(pipeline, pipeline_param, cv=5).fit(X_train, y_train)

print("Train Accuracy:", search.best_estimator_.score(X_train, y_train))
print("Test Accuracy:", search.best_estimator_.score(X_test, y_test))

#
# Save Model
#
# Data to be written
model_description ={
    "name" : "Miguel Chamochin",
    "title" : "Connecting the Sustainable Development Goals with climate change and energy transition",
    "tipo_ml" : "C",
    "target" : "Wealthy"
}

test_csv = pd.concat([X_test, y_test], axis=1)

dump_model(search.best_estimator_, 'Modelo_Pipeline_PCA_KNN_Cassification', model_description, test_csv)

loaded_model, X_test, y_test = load_model('Modelo_Pipeline_PCA_KNN_Cassification')

print('loaded_model.score', loaded_model.score(X_test, y_test))
