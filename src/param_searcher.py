# For params load
import yaml

# For data preprocessing
import numpy as np

# For ml operations
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit
import sklearn
# To manipulate paths to files
import os 
# To upload and download files
import pickle 
# Libraries for parsing
import argparse
import configparser
# Logging operations
from logger import Logger
# Preprocess yaml parameters (it is needed for random_search)
from utils import preprocess_params

parser = argparse.ArgumentParser(description="Predictor")

parser.add_argument( "-m",
                     "--model",
                     type=str,
                     help="Select model",
                     required=True,
                     choices=["log_reg", "rf", "svm"])

parser.add_argument( "-st",
                     "--search_type",
                     type=str,
                     help="Select search type",
                     required=True,
                     choices=["grid_search", "random_search"])                     

parser.add_argument("--category", "-cat",
                    type=str,
                    help="Choose the category to preprocess text for",
                    default="bestpicture",
                    choices=["bestpicture", "bestdirector", "bestacting"])

parser.add_argument("--verbose", "-v",
                    type=int,
                    help="Choose how many information to print",
                    default=3,
                    choices=[1, 2, 3])   

parser.add_argument("--num_iter", "-ni",
                    type=int,
                    help="Choose iteration number for random search",
                    default=100)

parser.add_argument("--random_state", "-rs",
                    type=int,
                    help="Random state for random search",
                    default=42)   

parser.add_argument("--scoring", "-s",
                    type=str,
                    help="Choose the scoring type Check types on https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)",
                    default="f1")     

parser.add_argument("--show_log", "-sl",
                    help="Choose whether to show logs",
                    action='store_true')                                    

parser.add_argument("--use_smote", "-us",
                    help="Choose whether to use smote to handle imbalances class",
                    action='store_true')   

parser.add_argument("--without_embeddings", "-we",
                    help="Get scaled df without text embeddings",
                    action='store_true')      

parser.add_argument("--save_parameters", "-sp",
                    help="Choose whether to save the best parameters",
                    action='store_true')                                     

args = parser.parse_args()                 

class ParamSearcher():

    def __init__(self):

        """
        Initialization paths for data and weights for future models
        """
        # Initialize logger and config
        logger = Logger(args.show_log)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.current_path = os.path.join(os.getcwd())
        self.config_path = os.path.join(self.current_path, "src", "config.ini")
        self.config.read(self.config_path)

        # Get estimator
        if args.model == 'log_reg':
            self.estimator = LogisticRegression()
        elif args.model == 'svm':
            self.estimator = SVC()
        else:
            self.estimator = RandomForestClassifier()
        # Get suffix for data path
        if args.without_embeddings: self.suffix="without_embeddings"
        else: self.suffix="with_embeddings"
        self.second_key = f"{args.category}_{self.suffix}"
        # The following 3 blocks load specified data (train and test)
        if args.use_smote:
            self.weights_prefix = "with_smote"
            self.train_path = self.config["SMOTE_DATA"][self.second_key]
            with open(self.train_path, 'rb') as file:
                smote_tuple_data = pickle.load(file)
            self.X_train, self.y_train =  smote_tuple_data
            print("DATA_WITH_SMOTE")

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
        # Upload params for searching from yaml
        self.param_searcher_path = os.path.join(self.current_path, "src", "model_params", "param_searcher.yaml")
        with open(self.param_searcher_path, "r") as file:
            self.params = yaml.safe_load(file)[args.search_type][args.model][f"{args.category}_params"]
        # Preprocess parameters
        self.params = preprocess_params(self.params, args.search_type)
        # Check parameters
        print(self.params)
        self.log.info("ParamSearcher is ready")

    def use_search_grid(self):

        """
        Thi function predefines splits and calcalutes GridSearch
        
        Returns:
        grid: grid that contains the best score and the best parameters
        """

        # Concatenate train and test data into one array
        x, y = np.concatenate((self.X_train, self.X_test), axis=0), np.concatenate((self.y_train, self.y_test), axis=0)
        # Specify the data indeces (1 for train and 0 for test)
        test_fold = [-1 for _ in range(self.X_train.shape[0])] + [0 for _ in range(self.X_test.shape[0])]
        ps = PredefinedSplit(test_fold)
        # Initialise grid search
        grid = GridSearchCV(self.estimator, self.params, refit = False, verbose = args.verbose, cv=ps, scoring=args.scoring)
        # fitting the model for grid search
        grid.fit(x, y)

        print(grid.best_params_)
        print(grid.best_score_)

        return grid

    def use_random_search(self):

        """
        This function predefines split and calculates RandomSearchCV
        
        Returns:
        random_search: random_search that contains the best score and the best parameters
        """

        # Concatenate train and test data into one array
        x, y = np.concatenate((self.X_train, self.X_test), axis=0), np.concatenate((self.y_train, self.y_test), axis=0)
        # Specify the data indeces (1 for train and 0 for test)
        test_fold = [-1 for _ in range(self.X_train.shape[0])] + [0 for _ in range(self.X_test.shape[0])]
        ps = PredefinedSplit(test_fold)
        random_search = RandomizedSearchCV(self.estimator, param_distributions=self.params, verbose = args.verbose,
                                           n_iter=args.num_iter, cv=ps, random_state=args.random_state, scoring=args.scoring)

        random_search.fit(x, y)
        # Print the best parameters and the best score
        print(random_search.best_params_)
        print(random_search.best_score_)

        return random_search

    def save_best_params(self, search):

        """
        This function saves the best parameters
        
        Parameters:
        search: search with the best parameters
        """

        # Get path and read the yaml
        main_param_path = os.path.join(self.current_path, "src", "model_params", "main_params.yaml")
        with open(main_param_path, "r") as file:
            main_params = yaml.safe_load(file)
        # Update the yaml with new parameters
        main_params[args.model][f"{args.category}_params"] = search.best_params_
        # Save updated yaml
        with open(main_param_path, 'w') as file:
            yaml.dump(main_params, file)

        self.log.info("The best parameters are saved")

if __name__ == "__main__":

    param_searcher = ParamSearcher()
    if args.search_type == "grid_search": 
        search = param_searcher.use_search_grid()
    else:
        search = param_searcher.use_random_search()

    if args.save_parameters:
        param_searcher.save_best_params(search)