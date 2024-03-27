# For data preprocessing
import pandas as pd
import numpy as np

# To get text embeddings with DistilBert
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

import os # to manipulate paths to files
import pickle # to upload and download files
import copy # to copy variables

# Libraries for parsing
import argparse
import configparser

# To log operations
from logger import Logger

from utils import create_config_dict

parser = argparse.ArgumentParser(description="Predictor")

parser.add_argument("--category", "-cat",
                    type=str,
                    help="Choose the category to preporcess text for",
                    default="bestpicture",
                    choices=["bestpicture", "bestdirector", "bestacting"])

parser.add_argument("--show_log", "-sl",
                    type=bool,
                    help="Choose whether to show logs",
                    action='store_true')                                   

parser.add_argument("--tokenizer_name", "-tn",
                    type=str,
                    help="Name for pretrained DistilBert tokenizer",
                    default="distilbert-base-uncased-finetuned-sst-2-english")

parser.add_argument("--model_name", "-mn",
                    type=str,
                    help="Name for pretrained DistilBert model",
                    default="distilbert-base-uncased-finetuned-sst-2-english")

parser.add_argument("--prediction", "-pr",
                    type=bool,
                    help="If False then use hidden state (tensor with more dims) to get embeddings",
                    action='store_true')

args = parser.parse_args()                    

class EmbedMaker():

    def __init__(self) -> None:
        """
        Initialization paths for data and weights for models
        """
        logger = Logger(args.show_log)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)

        self.current_path = os.path.join(os.getcwd(), 'src')
        self.review_folder = os.path.join(os.getcwd(), 'data/reviews', args.category)

        self.embeddings_folder = os.path.join(os.getcwd(), f'embeddings/{args.category}')
        os.makedirs(self.embeddings_folder, exist_ok=True)

        self.save_path = os.path.join(self.embeddings_folder, 'full_data_dict.pkl')
        self.rt_dict_path = os.path.join(self.review_folder, 'full_rt_dict.pkl')
        self.mc_dict_path = os.path.join(self.review_folder, 'full_mc_dict.pkl')
        self.df_path = os.path.join(os.getcwd(), 'data/dataframes', f'oscardata_{args.category}.csv')

        self.df = pd.read_csv(self.df_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained(args.tokenizer_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(args.model_name)
        self.model.to(self.device)

        self.log.info("DataMaker is ready")
        try:
            self.config.read('config.ini')
        except UnicodeDecodeError as er:
            self.config.read('config.ini', encoding='latin-1')

    def get_data(self) -> None:
        
        """
        This function applies reviews to the csv dataframe and then extract
        them as nested list for each films 
        """

        # Upload reviews as dicts
        with open(self.rt_dict_path, 'rb') as file:
            self.rt_dict = pickle.load(file)

        with open(self.mc_dict_path, 'rb') as file:
            self.mc_dict = pickle.load(file)

        # Add reviews to dataframe with reviews
        self.df['rt reviews'] = self.df['Film'].map(self.rt_dict)
        self.df['mc reviews'] = self.df['Film'].map(self.mc_dict)

        # Get reviews as lists if you are going to get embeddings
        rt_reviews = self.df['rt reviews']
        mc_reviews = self.df['mc reviews']

        self.log.info("Reviews were extracted")

        return rt_reviews, mc_reviews

    def get_hidden_state(self, reviews):
        """
        get hidden states
        """
        with torch.no_grad():
            # If there is nan instead of reviews thenreturn None
            if type(reviews) == type(np.nan):
                return None

            inputs = self.tokenizer(reviews, return_tensors="pt", padding='longest')
            inputs.to(self.device)
            distilbert_output = self.model.distilbert(**inputs)
            hidden_state = distilbert_output[0]
            hidden_state = hidden_state[:, 0]

            return hidden_state

    def get_prediction(self, hidden_state):
        """
        Get prediction if prediction parameter is True in get_embeddings function
        """
        with torch.no_grad():

            pooled_output = self.model.pre_classifier(hidden_state)  # (bs, dim)
            pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
            pooled_output = self.model.dropout(pooled_output)  # (bs, dim)
            logits = self.model.classifier(pooled_output)  # (bs, num_labels)

            return logits

    def get_embeddings(self, data, prediction=True):

        """
        This fucntion takes in data parameter that is dict and contains text reviews and add to it embeddings
        """

        # Copy data
        copied_data = copy.deepcopy(data)

        self.log.info("Preprocessing text to embedding started")

        # loop over review's dict {'mc_reviews':mc_reviews, 'rt_reviews':rt_reviews} that were uploaded recently in the notebook
        for j, (key, text_nested_list) in enumerate(copied_data.items()):
            # initialise lists
            hidden_states_list, prediction_list, sentiment_list = [], [], []

            print("-"*100)
            print(f"Preprocess {key} text")

            # iterate over reviews for each film
            for i, text_list in enumerate(text_nested_list):

                hidden_state = self.get_hidden_state(text_list)

                # Get embeddings as a prediction if prediction parameter is True
                if prediction:
                    if hidden_state != None:
                        logits = torch.mean(self.get_prediction(hidden_state), dim=0)
                        predicted_class_id = logits.argmax().item()
                        sentiment = self.model.config.id2label[predicted_class_id]

                        prediction_list.append(logits.detach().cpu().numpy())
                        sentiment_list.append(0 if sentiment == 'NEGATIVE' else 1)

                    else: prediction_list.append(None); sentiment_list.append(None)

                # Append hidden state
                if hidden_state != None: hidden_states_list.append(torch.mean(hidden_state, dim=0).detach().cpu().numpy())
                else: hidden_states_list.append(None)

                if i%100==0:
                    print("Number of iteration - ", i)

            # Add embeddings to data parameter
            data[key+'_hidden_states'] = hidden_states_list
            data[key+'_prediction_list'] = prediction_list
            data[key+'_sentiment_list'] = sentiment_list

        return data

    def preprocess_and_save(self):

        """
        This function retrieves text, preprocesses it and finally saves data
        """

        # Retrieve text as nested lists
        rt_reviews, mc_reviews = self.get_data()
        data = {'mc_reviews':mc_reviews, 'rt_reviews':rt_reviews}
        # Get embeddings
        data = self.get_embeddings(data, prediction=args.prediction)
        # Save embedings
        with open(self.save_path, 'wb') as file:
            pickle.dump(data, file)

        # Check config keys
        create_config_dict(self.config, keys=["EMBEDDINGS", "RAW_TEXT_REVIEWS", "CSV_FILE"])
        # Save paths to config.ini
        self.config["EMBEDDINGS"][f'{args.category}'] = self.save_path
        self.config["RAW_TEXT_REVIEWS"][f'{args.category}_rt_dict'] = self.rt_dict_path, 
        self.config["RAW_TEXT_REVIEWS"][f'{args.category}_mc_dict'] = self.mc_dict_path
        self.config["CSV_FILE"][f'{args.category}'] = self.df_path

        with open(os.path.join(self.current_path, "config.ini"), 'w') as configfile:
            self.config.write(configfile)

        self.log.info(f'Texts were preprocessed successful and embeds are saved at {self.review_folder}')

if __name__ == "__main__":
    embed_maker = EmbedMaker()
    # embed_maker.preprocess_and_save()
    # Save paths to config.ini
    embed_maker.config["EMBEDDINGS"] = {f'{args.category}':embed_maker.save_path}
    embed_maker.config["RAW_TEXT_REVIEWS"] = {f'{args.category}_rt_dict':embed_maker.rt_dict_path, f'{args.category}_mc_dict':embed_maker.mc_dict_path}
    embed_maker.config["CSV_FILE"] = {f'{args.category}':embed_maker.df_path}

    with open(os.path.join(embed_maker.current_path, "config.ini"), 'w') as configfile:
        embed_maker.config.write(configfile)

