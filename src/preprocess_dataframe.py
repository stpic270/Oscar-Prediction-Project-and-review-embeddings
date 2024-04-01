# For data preprocessing
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
# To manipulate paths to files
import os 
# To upload and download files
import pickle 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Import some utils for preprocessing dataframes
from utils import fill_in_mean_values, add_data_smote, create_config_dict
# Libraries for parsing
import argparse
import configparser
# Logging operations
from logger import Logger

parser = argparse.ArgumentParser(description="Predictor")

parser.add_argument("--use_hidden_states", "-hs",
                    help="Choose whether to use hidden_states as embeddings",
                    action='store_true')   

parser.add_argument("--use_smote", "-us",
                    help="Choose whether to use smote to handle imbalances class",
                    action='store_true')                            

parser.add_argument("--category", "-cat",
                    type=str,
                    help="Choose the category to preprocess text for",
                    default="bestpicture",
                    choices=["bestpicture", "bestdirector", "bestacting"])

parser.add_argument("--without_embeddings", "-we",
                    help="Get scaled df without text embeddings",
                    action='store_true')     

parser.add_argument("--show_log", "-sl",
                    help="Choose whether to show logs",
                    action='store_true')                                   

parser.add_argument("--random_state", "-rs",
                    type=int,
                    help="Random state for the split",
                    default=None)

parser.add_argument("--test_size", "-ts",
                    type=float,
                    help="Size for train split",
                    default=0.2)                    

args = parser.parse_args()    
 
class ScaleData():

    """
    This class scaled passed data
    """

    def __init__(self):
        # Initialize logger
        self.logger = Logger(args.show_log)
        self.config = configparser.ConfigParser()
        self.log = self.logger.get_logger(__name__) 
        # Read config
        self.current_path = os.path.join(os.getcwd())
        self.config_path = os.path.join(self.current_path, "src", "config.ini")
        self.config.read(self.config_path)
        # Get csv path and embeddings path
        self.csv_path = self.config["CSV_FILE"][args.category]
        self.embedding_path = self.config["EMBEDDINGS"][args.category]
        # Define whether to use embeddings or not
        if args.without_embeddings: self.suffix="without_embeddings"
        else: self.suffix="with_embeddings"
        # Folders path
        self.folder_path = os.path.join(self.current_path, "data", "scaled_dataframes", args.category)
        self.folder_suffix_path = os.path.join(self.folder_path, self.suffix)
        # Create Folders if needed
        os.makedirs(self.folder_path, exist_ok=True)
        os.makedirs(self.folder_suffix_path, exist_ok=True)
        # Define save paths
        self.smote_train_path = os.path.join(self.folder_suffix_path, "smote_train.pkl")
        self.usual_train_path = os.path.join(self.folder_suffix_path, "usual_train.pkl")
        self.test_path = os.path.join(self.folder_suffix_path, "test.pkl")

        self.log.info("ScaleData is ready")

    def get_df_with_embeddings(self):
        """
        This function uploads df, then applies embeddings, if there are nan values - fill in
        with average from column. It is also removes unused columns and extract labels
        
        Returns:
        df (pandas.DataFrame): final preprocessed dataframe
        y (list): labels for data (win oscar or not)
        """

        # Load csv file
        df_wt_reviews = pd.read_csv(self.csv_path)
        # Load our embeddings
        with open(self.embedding_path, 'rb') as file:
            data = pickle.load(file)
        # Retrieve lists with reviews
        rt_reviews_prediction_list, mc_reviews_prediction_list = data['rt_reviews_prediction_list'], data['mc_reviews_prediction_list']
        rt_reviews_sentiment_list, mc_reviews_sentiment_list = data['rt_reviews_sentiment_list'], data['mc_reviews_sentiment_list']
        # Fill in average values instead of None
        rt_reviews_prediction_list, mc_reviews_prediction_list = fill_in_mean_values(rt_reviews_prediction_list), fill_in_mean_values(mc_reviews_prediction_list)
        rt_reviews_sentiment_list, mc_reviews_sentiment_list = fill_in_mean_values(rt_reviews_sentiment_list), fill_in_mean_values(mc_reviews_sentiment_list)
        # Make prediction dataframes
        rt_prediction_df = pd.DataFrame(rt_reviews_prediction_list, columns=['rt_first_dim', 'rt_second_dim'])
        mc_prediction_df = pd.DataFrame(mc_reviews_prediction_list, columns=['mc_first_dim', 'mc_second_dim'])
        # Make sentiment dataframes
        rt_sentiment_df = pd.DataFrame(rt_reviews_sentiment_list, columns=['rt_sentiment'])
        mc_sentiment_df = pd.DataFrame(mc_reviews_sentiment_list, columns=['mc_sentiment'])
        # Get true labels and get rid of some columns 
        y = df_wt_reviews['Winner']
        df_wt_reviews = df_wt_reviews.drop(['Category', 'Film', 'Nominee', 'Year', 'Release_date', 'MPAA_rating', 'Winner'], axis=1)
        # Make one dataframe with embeddings
        df = pd.concat([df_wt_reviews, rt_prediction_df,
                        mc_prediction_df, rt_sentiment_df,
                        mc_sentiment_df], axis=1)

        if args.use_hidden_states:
            # Make the same operations and concatenate hidden states embeddings to the dataframe
            rt_hidden_states_list, mc_hidden_states_list = data['rt_reviews_hidden_states'], data['mc_reviews_hidden_states']
            rt_hidden_states_list, mc_hidden_states_list = fill_in_mean_values(rt_hidden_states_list), fill_in_mean_values(mc_hidden_states_list)
            # Get shape in order to convert tensor to the dataframe columns
            d = mc_hidden_states_list[0].shape[0]
            rt_df = pd.DataFrame(rt_hidden_states_list, columns=[f'rt_embed_index_{i}' for i in range(d)])
            mc_df = pd.DataFrame(mc_hidden_states_list, columns=[f'mc_embed_index_{i}' for i in range(d)])
            df = pd.concat([df, rt_df, mc_df], axis=1)

            self.log.info("Hidden states are added to the dataframe successful")

        return df, y


    def scale_data(self, df, y):
        """
        This function scales and saves scaled data
        
        Parameters:
        df (pandas.DataFrame): dataframe from get_df_with_embeddings function
        y(list): labels from get_df_with_embeddings function
        """
        # Scale data
        df = StandardScaler().fit_transform(df)
        # Get train, test split
        X_train, X_test, y_train, y_test = train_test_split(df,y,test_size=args.test_size,
                                                    random_state=args.random_state)

        usual_tuple_data = (X_train, y_train)
        test_tuple_data = (X_test, y_test)
        # Save data
        with open(self.usual_train_path, 'wb') as file:
                pickle.dump(usual_tuple_data, file)
        
        with open(self.test_path, 'wb') as file:
                pickle.dump(test_tuple_data, file)

        # Check config keys
        create_config_dict(self.config, keys=["USUAL_DATA", "TEST_DATA", "SMOTE_DATA"])
        # Define paths for config
        self.config["USUAL_DATA"][f"{args.category}_{self.suffix}"] = self.usual_train_path
        self.config["TEST_DATA"][f"{args.category}_{self.suffix}"] = self.test_path
        
        self.log.info("Your data was scaled and saved successful")

        if args.use_smote:
            counter = Counter(y_train)
            # Use SMOTE
            art_X_train, art_y_train = add_data_smote(X_train, y_train, random_state=args.random_state)
            art_counter = Counter(art_y_train)
            # Check the size of train split after SMOTE
            print("Before SMOTE embeddings data", counter)
            print("After SMOTE embeddings data", art_counter)

            smote_tuple_data = (art_X_train, art_y_train)
            # Save SMOTE data
            with open(self.smote_train_path, 'wb') as file:
                pickle.dump(smote_tuple_data, file)

            # Save paths to config.ini
            self.config["SMOTE_DATA"][f"{args.category}_{self.suffix}"] = self.smote_train_path

            self.log.info("SMOTE data was created and saved successful")
            
        with open(self.config_path, 'w') as configfile:
                self.config.write(configfile)

if __name__ == "__main__":

    scaler = ScaleData()

    if args.without_embeddings: 
        df = pd.read_csv(scaler.csv_path)
        y = df['Winner']
        df = df.drop(['Category', 'Film', 'Nominee', 'Year', 'Release_date', 'MPAA_rating', 'Winner'], axis=1)

    else: df, y = scaler.get_df_with_embeddings()

    scaler.scale_data(df, y)
        