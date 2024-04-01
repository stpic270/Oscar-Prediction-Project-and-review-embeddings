# For params load
import yaml
# For data preprocessing
import pandas as pd
# For ml operations
from sklearn.metrics import confusion_matrix, classification_report
# For data plotting
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
# To manipulate paths to files
import os 
# To upload and download files
import pickle
# Libraries for parsing
import argparse
import configparser
# Logging operations
from logger import Logger

parser = argparse.ArgumentParser(description="Predictor")

parser.add_argument("--use_smote", "-us",
                    type=str,
                    help="Choose whether to use model on smote data or usual data",
                    required=True,
                    choices=["True", "False"])  

parser.add_argument("--category", "-cat",
                    type=str,
                    help="Choose the category to preprocess text for",
                    default="bestpicture",
                    choices=["bestpicture", "bestdirector", "bestacting"])

parser.add_argument("--show_log", "-sl",
                    help="Choose whether to show logs",
                    action='store_true')         

parser.add_argument("--delete_png", "-dp",
                    help="Choose whether to save png and pdf or just pdf",
                    action='store_true')                                                                        

args = parser.parse_args()                 

class Comparer():
    def __init__(self):

        """
        Initialization paths for data and weights for future models
        """

        # Define prefix
        if args.use_smote == "True": self.prefix = "with_smote"
        else: self.prefix = "without_smote" 
        # Initialize logger and config
        logger = Logger(args.show_log)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.current_path = os.path.join(os.getcwd())
        self.config_path = os.path.join(self.current_path, "src", "config.ini")
        self.config.read(self.config_path)
        # Upload class names and specified models from yaml
        self.names_path = os.path.join(self.current_path, "src", "model_params", "names.yaml")
        with open(self.names_path, "r") as file:
            yaml_file = yaml.safe_load(file)
            self.model_names = yaml_file["names"]["model_names"]
            self.class_names = yaml_file["names"]["class_names"]
        # Specified folder paths (create folders if needed)
        self.category_path = os.path.join(self.current_path, "src", "experiments", "graphics","model_compare", args.category)
        self.smote_path = os.path.join(self.category_path, self.prefix + "_" + "-".join(self.model_names))
        os.makedirs(self.category_path, exist_ok=True)
        os.makedirs(self.smote_path, exist_ok=True)
        # Specify suffix
        self.suffix_with_embeddings = f"{args.category}_with_embeddings"
        self.suffix_without_embeddings = f"{args.category}_without_embeddings"
        # Upload test data (with embeddings and without them)
        with open(self.config["TEST_DATA"][self.suffix_with_embeddings], 'rb') as file:
            test_tuple_data = pickle.load(file)
            self.X_test_with_embeddings, self.y_test_with_embeddings  = test_tuple_data
        with open(self.config["TEST_DATA"][self.suffix_without_embeddings], 'rb') as file:
            test_tuple_data = pickle.load(file)
            self.X_test_without_embeddings, self.y_test_without_embeddings  = test_tuple_data
        # Create dict with models 
        self.models_dict = {}
        for name in self.model_names:

            with open (self.config[name][f"{args.category}_with_embeddings"], "rb") as file:
                self.models_dict[f"{name}_with_embeddings"] = pickle.load(file)

            with open (self.config[name][f"{args.category}_without_embeddings"], "rb") as file:
                self.models_dict[f"{name}_without_embeddings"] = pickle.load(file)

        self.log.info("Predictor is ready")

    def get_graphics(self):
        
        """
        This function creates confusion matrix and classification report for each models
        
        Returns:
        cms (list): list with confusion matrices
        reports (list): list with classification reports
        """

        reports, cms = [], []
        for model_name in self.model_names:
            # Extract models
            key_1, key_2 = f"{model_name}_with_embeddings", f"{model_name}_without_embeddings"
            model_1 = self.models_dict[key_1]
            model_2 = self.models_dict[key_2]
            # Get prediction and classification report for the first model
            y_pred_1 = model_1.predict(self.X_test_with_embeddings)
            report_1 = classification_report(self.y_test_with_embeddings, y_pred_1,
                                            target_names=self.class_names, output_dict=True)

            # Get prediction and calssification report for the second model
            y_pred_2 = model_2.predict(self.X_test_without_embeddings)
            report_2 = classification_report(self.y_test_without_embeddings,y_pred_2,
                                            target_names=self.class_names, output_dict=True)

            # Get confusion matrices
            cm1 = confusion_matrix(self.y_test_with_embeddings, y_pred_1)
            cm2 = confusion_matrix(self.y_test_without_embeddings ,y_pred_2)
            # Apeend results
            reports.append((report_1, key_1)); reports.append((report_2,key_2))
            cms.append((cm1,key_1)); cms.append((cm2, key_2))

        return reports, cms

    def plot_reports(self, reports):

        """
        This function plots classification repors and saves results

        Parameters:
        reports (list): list with classification reports
        """

        for i in range(0, len(reports), 2):
            
            fig, axs = plt.subplots(2, 1, figsize=(5, 5))
            (report_1, name_1), (report_2, name_2) = reports[i], reports[i+1]
            j = i // 2
            # Convert classification reports to pandas DataFrames for easier manipulation
            df_1 = pd.DataFrame(report_1).T
            df_2 = pd.DataFrame(report_2).T

            # Reset index for DataFrames
            df_1.reset_index(inplace=True)
            df_2.reset_index(inplace=True)

            # Set index name for the DataFrames
            df_1.rename(columns={'index': 'Metric'}, inplace=True)
            df_2.rename(columns={'index': 'Metric'}, inplace=True)

            # Create a figure and axis
            fig, ax = plt.subplots(2, 1, figsize=(20, 5))

            # Plot the classification reports as tables
            ax[0].axis('off')
            ax[0].table(cellText=df_1.values, colLabels=df_1.columns, cellLoc='center', loc='center')
            ax[0].set_title(name_1)

            ax[1].axis('off')
            ax[1].table(cellText=df_2.values, colLabels=df_2.columns, cellLoc='center', loc='center')
            ax[1].set_title(name_2)

            # Save each figure as an image
            fig.savefig(os.path.join(f"{self.smote_path}", f"report_{j}.png"))
            plt.close(fig)
        # Create pdf with classification reports
        c = canvas.Canvas(os.path.join(f"{self.smote_path}", "multiple_reports.pdf"))
        for i in range(len(reports)//2):
            c.drawImage(os.path.join(f"{self.smote_path}", f"report_{i}.png"), 10, 300, width=600, height=350)
            c.showPage()
        # Save pdf report
        c.save()

    def plot_cms(self, cms):

        """
        This function plots confusion matrices and saves results

        Parameters:
        cms(list): list with confusion matrices
        """

        for i in range(0, len(cms), 2):
            
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            j = i // 2
            (cm1, name_1), (cm2, name_2)  = cms[i], cms[i+1]

            # Plot confusion matrix 1
            sns.heatmap(cm1, annot=True, cmap='Blues', fmt='d', cbar=False, ax=axs[0])
            axs[0].set_title(name_1)
            axs[0].set_xlabel('Predicted Label')
            axs[0].set_ylabel('True Label')
            axs[0].set_xticklabels(self.class_names)
            axs[0].set_yticklabels(self.class_names)

            # Plot confusion matrix 2
            sns.heatmap(cm2, annot=True, cmap='Oranges', fmt='d', cbar=False, ax=axs[1])
            axs[1].set_title(name_2)
            axs[1].set_xlabel('Predicted Label')
            axs[1].set_ylabel('True Label')
            axs[1].set_xticklabels(self.class_names)
            axs[1].set_yticklabels(self.class_names)

            # Save each figure as an image
            fig.savefig(os.path.join(f"{self.smote_path}", f"cm_{j}.png"))
            plt.close(fig)
        # Create pdf
        c = canvas.Canvas(os.path.join(f"{self.smote_path}", "multiple_cms.pdf"))
        for i in range(len(cms)//2):
            c.drawImage(os.path.join(f"{self.smote_path}", f"cm_{i}.png"), 30, 300, width=600, height=300)
            c.showPage()
        # Save pdf
        c.save()

    def delete_png(self, name, length):

        """
        This function deletes created png images
        
        Parameters:
        name (str): name for file (cm (confusion matrix) or cr (classification report))
        length (int): number of png files for cr and cm 
        """

        for i in range(length):
            
            path = os.path.join(self.smote_path, f"{name}_{i}.png")
            os.remove(path)        

if __name__ == "__main__":

    comparer = Comparer()
    reports, cms = comparer.get_graphics()
    comparer.plot_reports(reports)
    comparer.plot_cms(cms)

    comparer.log.info("PDF with results was create")

    if args.delete_png:
        comparer.delete_png(name="report", length=len(reports)//2)
        comparer.delete_png(name="cm", length=len(cms)//2) 
        comparer.log.info("PNGs were deleted")   
