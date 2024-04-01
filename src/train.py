# For params load
import yaml
# For ml operations
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import sklearn
# To manipulate paths to files
import os 
# To upload and download files
import pickle 
# Log exception
import traceback
# For exit if exception is caught
import sys
# Libraries for parsing
import argparse
import configparser
# To log operations
from logger import Logger
# For creating config
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
        # Get logger and read the config
        logger = Logger(args.show_log)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.current_path = os.path.join(os.getcwd())
        self.config_path = os.path.join(self.current_path, "src", "config.ini")
        self.config.read(self.config_path)
        # Define suffix in order to specify which data to use
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
        # Create folders for saving models
        self.experiments_category = os.path.join(self.current_path, "src", "experiments", args.category)
        self.model_weights_path = os.path.join(self.experiments_category, self.suffix)
        os.makedirs(self.experiments_category, exist_ok=True)
        os.makedirs(self.model_weights_path , exist_ok=True)
        # Specify path for saving
        self.save_path = os.path.join(self.model_weights_path, f"{self.weights_prefix}_{args.model}.pickle")
        # Specify path to yaml file
        self.params_path = os.path.join(self.current_path, "src", "model_params", "main_params.yaml")
        # Upload parameters from yaml
        with open(self.params_path, "r") as file:
            self.params = yaml.safe_load(file)[args.model][f"{args.category}_params"]

        self.log.info("MultiModel is ready")

    def train_model(self):

        """
        This model load model and return it
        
        Returns:
        classifier: sklearn model
        """

        # Upload model
        if args.model == 'log_reg':
            classifier = LogisticRegression(**self.params)
        elif args.model == 'svm':
            classifier = SVC(**self.params)
        else:
            classifier = RandomForestClassifier(**self.params)
        # Fit model (check parmaeters compatibilities)
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        # Calculate accuracy if needed
        if args.predict:
            y_pred = classifier.predict(self.X_test)
            print('Accuracy on test data - ',accuracy_score(self.y_test, y_pred))

        return classifier

    def save_model(self, classifier) -> bool:

        """
        This function saves the model
        
        Parameters:
        classifier: sklearn model to save
        """

        # Save trained models
        with open(self.save_path, 'wb') as f:
            pickle.dump(classifier, f)

        # Check config keys
        create_config_dict(self.config, keys=[args.model])
        self.config[args.model][f"{args.category}_{self.suffix}"]=self.save_path 
        # Update config
        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)

        self.log.info(f'{args.model} model is saved at f{self.save_path}')

if __name__ == "__main__":

    multimodel = MultiModel()
    classifier = multimodel.train_model()
    multimodel.save_model(classifier)


   




