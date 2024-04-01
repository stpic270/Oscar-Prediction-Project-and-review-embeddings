# For data preprocessing
import pandas as pd
# For ml operations
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# For data plotting
import plotly.io as pio
import plotly.express as px
import plotly
import tqdm
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

parser.add_argument("--try_neigbours", "-tn",
                    type=int,
                    help="Choose the number of neigbours",
                    default=50)     
                    
parser.add_argument("--show_graph_in_terminal", "-sg",
                    help="Choose whether to show graph and save it or just save it",
                    action='store_true')                                                                         

args = parser.parse_args()                 

class Vizualizater():
    def __init__(self):
        
        """
        Initialization paths for data and weights for future models
        """

        # Define prefix for data path
        if args.use_smote == "True": self.prefix = "with_smote"
        else: self.prefix = "without_smote" 
        # Initialize logger and config
        logger = Logger(args.show_log)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.current_path = os.path.join(os.getcwd())
        self.config_path = os.path.join(self.current_path, "src", "config.ini")
        self.config.read(self.config_path)
        # Define path for data and create folders if needed
        self.category_path = os.path.join(self.current_path, "src", "experiments", "graphics","data_analysis", args.category)
        self.smote_path = os.path.join(self.category_path, self.prefix)
        os.makedirs(self.category_path, exist_ok=True)
        os.makedirs(self.smote_path, exist_ok=True)
        # Define suffix 
        self.suffix_with_embeddings = f"{args.category}_with_embeddings"
        self.suffix_without_embeddings = f"{args.category}_without_embeddings"
        # Load data with and without embedding using smote
        if args.use_smote == "True":
            with open(self.config["SMOTE_DATA"][self.suffix_with_embeddings], 'rb') as file:
                test_tuple_data = pickle.load(file)
                self.X_train_with_embeddings, self.y_train_with_embeddings  = test_tuple_data
            with open(self.config["SMOTE_DATA"][self.suffix_without_embeddings], 'rb') as file:
                test_tuple_data = pickle.load(file)
                self.X_train_without_embeddings, self.y_train_without_embeddings = test_tuple_data

        # Load data with and without embedding not using smote
        else:
            with open(self.config["USUAL_DATA"][self.suffix_with_embeddings], 'rb') as file:
                test_tuple_data = pickle.load(file)
                self.X_train_with_embeddings, self.y_train_with_embeddings  = test_tuple_data
            with open(self.config["USUAL_DATA"][self.suffix_without_embeddings], 'rb') as file:
                test_tuple_data = pickle.load(file)
                self.X_train_without_embeddings, self.y_train_without_embeddings = test_tuple_data

        self.log.info("Visualizer is ready")

    def compute_tsne(self, scaled_data_1, scaled_data_2,
                     n_comp_1=2, n_comp_2=2):

        """
        This function computes TSNE for data

        Parameters:
        scaled_data_1: Data for the first graphic
        scaled_data_2: Data for the second graphic
        n_comp_1 (int): Dimension to which shrink the first data
        n_comp_2 (int): Dimension to which shrink the second data

        Returns:
        tuple: pair of trained TSNE
        """

        # Use TSNE to reduce dimensionality for the first data
        tsne_model_1 = TSNE(n_components=n_comp_1, random_state=42)
        tsne_values_1 = tsne_model_1.fit_transform(scaled_data_1)

        # Use TSNE to reduce dimensionality for the second data
        tsne_model_2 = TSNE(n_components=n_comp_2, random_state=42)
        tsne_values_2 = tsne_model_2.fit_transform(scaled_data_2)

        self.log.info("TSNE are computed")

        return tsne_values_1, tsne_values_2

    def plot_2_TSNE(self, tsne_1 , tsne_2,
                    y_1, y_2,
                    title_1 = 't-SNE WITH embeddings',
                    title_2 = 't-SNE WITHOUT embeddings'
                    ):

        """
        This function takes TSNE values from compute_tsne function and plots the results

        Parameters:
        tsne_1: Data for the first graphic
        tsne_2: Data for the second graphic
        y_1: labels
        y_2: labels
        title_1 (str): Title for the first data
        title_2 (str): Title for the second data 
        """

        # Set the first graphic
        fig1 = px.scatter(
        x = tsne_1[:,0],
        y = tsne_1[:,1],
        color = y_1,
        title = title_1, width = 800, height = 600,
        color_discrete_sequence = plotly.colors.qualitative.Alphabet_r
        )

        # Set the second graphic
        fig2 = px.scatter(
        x = tsne_2[:,0],
        y = tsne_2[:,1],
        color = y_2,
        title = title_2, width = 800, height = 600,
        color_discrete_sequence = plotly.colors.qualitative.Alphabet_r
        )

        # Change the titles for both graphics
        fig1.update_layout(
            xaxis_title = 'first component',
            yaxis_title = 'second component')

        fig2.update_layout(
            xaxis_title = 'first component',
            yaxis_title = 'second component')

        if args.show_graph_in_terminal:
            fig1.show()
            fig2.show()
        # Define paths for images
        path1 = os.path.join(f"{self.smote_path}", f"tsne_with_embeddings.png")
        path2 = os.path.join(f"{self.smote_path}", f"tsne_without_embeddings.png")
        # Write images
        pio.write_image(fig1, path1)
        pio.write_image(fig2, path2)

        return [path1, path2]

    def k_means_neigbours_graph(self, scaled_data_1, scaled_data_2, 
                                title_1="neigbours_with_embeddings", 
                                title_2="neigbours_without_embeddings"):

        """
        This function calculates k-means with different amount of k, computes
        silhouette score and plots the result

        Parameters:
        scaled_data_1: Data for the first k-means
        scaled_data_2: Data for the second k-means

        title_1 (str): Title for the first data
        title_2 (str): Title for the second data 
        """

        # Initialise lists
        silhouette_scores_1, silhouette_scores_2 = [], []

        # Loop to compute k-means and silhouette score for the first data
        for k in tqdm.tqdm(range(2, args.try_neigbours)):
            kmeans = KMeans(n_clusters=k,
                            random_state=42,
                            n_init = 'auto').fit(scaled_data_1)
            kmeans_labels = kmeans.labels_
            silhouette_scores_1.append(
                {
                    'k': k,
                    'silhouette_score': silhouette_score(scaled_data_1,
                        kmeans_labels, metric = 'cosine')
                }
            )

        # Loop to compute k-means and silhouette score for the second data
        for k in tqdm.tqdm(range(2, args.try_neigbours)):
            kmeans = KMeans(n_clusters=k,
                            random_state=42,
                            n_init = 'auto').fit(scaled_data_2)
            kmeans_labels = kmeans.labels_
            silhouette_scores_2.append(
                {
                    'k': k,
                    'silhouette_score': silhouette_score(scaled_data_2,
                        kmeans_labels, metric = 'cosine')
                }
            )

        # Set graphics
        fig1 = px.line(pd.DataFrame(silhouette_scores_1).set_index('k'),
            title = f'<b>The best number of classes for {title_1}</b>',
            labels = {'value': 'silhoutte score'},
            color_discrete_sequence = plotly.colors.qualitative.Alphabet)
        fig1.update_layout(showlegend = False)

        fig2 = px.line(pd.DataFrame(silhouette_scores_2).set_index('k'),
            title = f'<b>The best number of classes for {title_2}</b>',
            labels = {'value': 'silhoutte score'},
            color_discrete_sequence = plotly.colors.qualitative.Alphabet)
        fig2.update_layout(showlegend = False)

        if args.show_graph_in_terminal:
            fig1.show()
            fig2.show()
        # Define save paths
        path1 = os.path.join(f"{self.smote_path}", f"{title_1}.png")
        path2 = os.path.join(f"{self.smote_path}", f"{title_2}.png")
        # Save images
        pio.write_image(fig1, path1)
        pio.write_image(fig2, path2)

        self.log.info("K-means-neigbours graph was calculated")

        return [path1, path2]

    def save_pdf(self, tsne_path, neigbours_path):

        """
        This function creates pdf
        
        Parameters:
        tsne_path (str): path for tsne image
        neigbours_path (str): path for k-neigbours image
        """

        # Initialize pdf
        c = canvas.Canvas(os.path.join(f"{self.smote_path}", "analysed_data.pdf"))
        # Write images
        for path in tsne_path:
            c.drawImage(path , 60, 300, width=500, height=400)
            c.showPage()

        for path in neigbours_path:
            c.drawImage(path , 30, 300, width=600, height=300)
            c.showPage()
        # Save pdf
        c.save()

        self.log.info("PDF is saved")
        # Delete png if specified
        if args.delete_png:
            tsne_path.extend(neigbours_path)

            for path in tsne_path:
                os.remove(path)

            self.log.info("PNGs were deleted")

if __name__ == "__main__":

    visualizater = Vizualizater()
    tsne_1, tsne_2 = visualizater.compute_tsne(visualizater.X_train_with_embeddings, 
                                               visualizater.X_train_without_embeddings)

    tsne_path = visualizater.plot_2_TSNE(tsne_1, tsne_2, 
                                         visualizater.y_train_with_embeddings, 
                                         visualizater.y_train_without_embeddings)

    neigbours_path = visualizater.k_means_neigbours_graph(visualizater.X_train_with_embeddings, 
                                                          visualizater.X_train_without_embeddings)                             

    visualizater.save_pdf(tsne_path, neigbours_path)