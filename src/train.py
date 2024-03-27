# For params load
import yaml

# For data preprocessing
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter

# For ml operations
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
import sklearn

# For data plotting
import plotly.express as px
import plotly
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import os # to manipulate paths to files
import time # to check time
import pickle # to upload and download files
import copy # to copy variables
import traceback
import sys

import argparse
import configparser
from logger import Logger

from utils import create_config_dict

parser = argparse.ArgumentParser(description="Predictor")

parser.add_argument( "-m",
                     "--model",
                     type=str,
                     help="Select model",
                     required=True,
                     choices=["log_reg", "rf", "svm"])

parser.add_argument("--category", "-cat",
                    type=str,
                    help="Choose the category to preprocess text for",
                    default="bestpicture",
                    choices=["bestpicture", "bestdirector", "bestacting"])

parser.add_argument("--show_log", "-sl",
                    help="Choose whether to show logs",
                    action='store_true')                                    

parser.add_argument("--use_smote", "-us",
                    help="Choose whether to use smote to handle imbalances class",
                    action='store_true')  

parser.add_argument("--predict", "-pr",
                    help="Check accuracy on test data or not",
                    action='store_true')          

parser.add_argument("--without_embeddings", "-we",
                    help="Get scaled df without text embeddings",
                    action='store_true')                 

args = parser.parse_args()                 

class MultiModel():

    def __init__(self):

        """
        Initialization paths for data and weights for future models
        """

        logger = Logger(args.show_log)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.current_path = os.path.join(os.getcwd())
        self.config_path = os.path.join(self.current_path, "src", "config.ini")
        self.config.read(self.config_path)

        if args.without_embeddings: self.suffix="without_embeddings"
        else: self.suffix="with_embeddings"
        self.second_key = f"{args.category}_{self.suffix}"

        if args.use_smote:
            
            self.weights_prefix = "with_smote"
            self.train_path = self.config["SMOTE_DATA"][self.second_key]
            with open(self.train_path, 'rb') as file:
                smote_tuple_data = pickle.load(file)

            self.X_train, self.y_train =  smote_tuple_data

        else:
            
            self.weights_prefix = "without_smote"
            self.train_path = self.config["USUAL_DATA"][self.second_key]
            with open(self.train_path, 'rb') as file:
                usual_tuple_data = pickle.load(file)

            self.X_train, self.y_train =  usual_tuple_data

        self.test_path = self.config["TEST_DATA"][self.second_key]
        with open(self.test_path, 'rb') as file:
                test_tuple_data = pickle.load(file)

        self.X_test, self.y_test = test_tuple_data

        self.experiments_category = os.path.join(self.current_path, "src", "experiments", args.category)
        self.model_weights_path = os.path.join(self.experiments_category, self.suffix)
        os.makedirs(self.experiments_category, exist_ok=True)
        os.makedirs(self.model_weights_path , exist_ok=True)

        self.save_path = os.path.join(self.model_weights_path, f"{self.weights_prefix}_{args.model}.pickle")
        self.params_path = os.path.join(self.current_path, "src", "model_params", "params.yaml")

        with open(self.params_path, "r") as file:
            self.params = yaml.safe_load(file)[args.model][f"{args.category}_params"]

        self.log.info("MultiModel is ready")

    def train_model(self):

        if args.model == 'log_reg':
            classifier = LogisticRegression(**self.params)
        elif args.model == 'svm':
            classifier = SVC(**self.params)
        else:
            classifier = RandomForestClassifier(**self.params)
        
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        if args.predict:
            y_pred = classifier.predict(self.X_test)
            print('Accuracy on test data - ',accuracy_score(self.y_test, y_pred))

        return classifier

    def save_model(self, classifier) -> bool:
        # Save trained models
        with open(self.save_path, 'wb') as f:
            pickle.dump(classifier, f)

        # Check config keys
        create_config_dict(self.config, keys=[args.model])
        self.config[args.model][f"{args.category}_{self.suffix}"]=self.save_path 

        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)

        print(self.config.keys)
        self.log.info(f'{args.model} model is saved at f{self.save_path}')

if __name__ == "__main__":

    multimodel = MultiModel()
    classifier = multimodel.train_model()
    multimodel.save_model(classifier)


   




