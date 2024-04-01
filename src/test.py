# For params load
import yaml
# For data preprocessing
import pandas as pd
# For ml operations
from sklearn.metrics import confusion_matrix, classification_report
# For data plotting
import seaborn as sns
import matplotlib.pyplot as plt
# To manipulate paths to files
import os  
# To upload and download files
import pickle 
# Libraries for parsing
import argparse
import configparser
# To log operations
from logger import Logger

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

parser.add_argument("--without_embeddings", "-we",
                    help="Get scaled df without text embeddings",
                    action='store_true')    

parser.add_argument("--save_grahics", "-sg",
                    help="Choose whether to save graphics",
                    action='store_true')             

args = parser.parse_args()                 

class Predictor():

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
        # Define key for test path
        if args.without_embeddings: self.suffix="without_embeddings"
        else: self.suffix="with_embeddings"
        self.second_key = f"{args.category}_{self.suffix}"
        # Upload test data
        self.test_path = self.config["TEST_DATA"][self.second_key]
        with open(self.test_path, 'rb') as file:
                test_tuple_data = pickle.load(file)
        self.X_test, self.y_test = test_tuple_data
        # Upload classifier
        self.classifier_path = self.config[args.model][f"{args.category}_{self.suffix}"]
        with open(self.classifier_path, "rb") as file:
            self.classifier = pickle.load(file)
        # Define path for class names (from yaml file)
        self.params_path = os.path.join(self.current_path, "src", "model_params", "names.yaml")
        # Get class names from yaml
        with open(self.params_path, "r") as file:
            self.class_names = yaml.safe_load(file)["names"]["class_names"]
        # Get a name after the last slash 
        after_slash = self.classifier_path.split('/')[-1]
        # Get title suffix (with smote/without smote)
        title_suffix = ' '.join(after_slash.split("_")[0:2])
        # Get title
        self.title = f"{args.model} {self.suffix.replace('_', ' ')} {title_suffix}"
        # Define paths for saving confusion matrix and classification report
        if args.save_grahics:
            self.graphics_categgory_path = os.path.join(self.current_path, "src", "experiments", "graphics", "test", args.category)
            self.graphics_category_embeddings_path = os.path.join(self.graphics_categgory_path, self.suffix)
            os.makedirs(self.graphics_categgory_path, exist_ok=True)
            os.makedirs(self.graphics_category_embeddings_path, exist_ok=True)
            self.cm_save_path = os.path.join(self.graphics_category_embeddings_path, f"cm_{args.model}_{title_suffix.replace(' ', '_')}.png")
            self.cr_save_path = os.path.join(self.graphics_category_embeddings_path, f"cr_{args.model}_{title_suffix.replace(' ', '_')}.png")

        self.log.info("Predictor is ready")

    def predict(self):

        """
        This function makes predictions on test data and calculate classification report
        and confusion report 

        Returns:
        report: classification report
        cm: confusion matrix
        """

        # Get prediction and classification report for the first model
        self.y_pred = self.classifier.predict(self.X_test)
        report = classification_report(self.y_test, self.y_pred,
                                       target_names=self.class_names, output_dict=True)
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test,self.y_pred)

        self.log.info("Predictions are ready")

        return report, cm

    def plot_report(self, report):
        
        """
        This function plots classification report

        Parameters:
        report: classification report from predict function
        """

        # Convert classification reports to pandas DataFrames for easier manipulation
        df = pd.DataFrame(report).T

        # Reset index for DataFrames
        df.reset_index(inplace=True)

        # Set index name for the DataFrames
        df.rename(columns={'index': 'Metric'}, inplace=True)

        # Create a figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(20, 5))

        # Plot the classification reports as tables
        ax.axis('off')
        ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        ax.set_title(self.title)
        # Save classification report if needed
        if args.save_grahics:
            plt.savefig(self.cr_save_path)

        plt.show()

    def plot_confusion_matrix(self, cm):

        """
        This function plots confusion matrix

        Parameters:
        report: confusion matrix from predict function
        """
        # Create a figure and axis
        fig, axs = plt.subplots(1, 1, figsize=(12, 6))

        # Plot confusion matrix 
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False, ax=axs)
        axs.set_title(self.title)
        axs.set_xlabel('Predicted Label')
        axs.set_ylabel('True Label')
        axs.set_xticklabels(self.class_names)
        axs.set_yticklabels(self.class_names)

        plt.tight_layout()
        # Save confusion matrix if needed
        if args.save_grahics:
            plt.savefig(self.cm_save_path)

            self.log.info("Graphics are saved")

        plt.show()

if __name__ == "__main__":

    predictor = Predictor()
    report, cm = predictor.predict()
    predictor.plot_report(report)
    predictor.plot_confusion_matrix(cm)