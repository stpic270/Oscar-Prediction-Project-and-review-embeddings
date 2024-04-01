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
from scipy.stats import randint, uniform

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

def fill_in_mean_values(data_list):

    """
    This function takes in data_list parameter, calculates the average variable for it and fill in average
    """
    temporary_list = []

    # Use it to avoid get None value as a type for the whole list
    temporary_type, i = type(None), 0
    while temporary_type == type(None):
        temporary_type = type(data_list[i])
        i += 1
    str_type = str(temporary_type)

    # Get not nan values
    for i in range(len(data_list)):
        if str(data_list[i]) != 'None':
            temporary_list.append(data_list[i])

    # form np arrays
    temporary_list_numpy = np.array(temporary_list)

    # Get means and type
    mean_tensor = np.mean(temporary_list_numpy, axis=0)
    if 'int' in str_type:
        mean_tensor = np.round(mean_tensor).astype(int)

    # Fill in means
    data_list = [arr if type(None) != type(arr) else mean_tensor for arr in data_list]

    return data_list

def add_data_smote(X, y, random_state):

    """
    This function adds data to imbalanced class

    Parameters:
    X: Data to which will be sampled new data
    y: Labels for data, new labels also will be sampled

    Returns:
    Dataset (tuple) with new data for imbalanced class
    """

    smote = SMOTE(sampling_strategy='auto', random_state=random_state)
    X_art, y_art = smote.fit_resample(X,y)

    return X_art, y_art

def create_config_dict(config, keys):
    for key in keys:
        if key not in config: config[key] = {}

def plot_2_cm(cm1, cm2, name_1, name_2, class_names):

    """
    This function takes parameters for plotting 2 confusion matrices and compare them

    Parameters:
    cm1: The first confusion matrix
    cm2: The second confusion matrix
    name_1 (str): Name for the cm1 on the plot
    name_2 (str): Name for the cm2 on the plot
    class_names (list): Names for the classes
    """

    # Create subplots for confusion matrices side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot confusion matrix 1
    sns.heatmap(cm1, annot=True, cmap='Blues', fmt='d', cbar=False, ax=axs[0])
    axs[0].set_title(name_1)
    axs[0].set_xlabel('Predicted Label')
    axs[0].set_ylabel('True Label')
    axs[0].set_xticklabels(class_names)
    axs[0].set_yticklabels(class_names)

    # Plot confusion matrix 2
    sns.heatmap(cm2, annot=True, cmap='Oranges', fmt='d', cbar=False, ax=axs[1])
    axs[1].set_title(name_2)
    axs[1].set_xlabel('Predicted Label')
    axs[1].set_ylabel('True Label')
    axs[1].set_xticklabels(class_names)
    axs[1].set_yticklabels(class_names)

    plt.tight_layout()
    plt.show()

def plot_2_reports(report_1, report_2, name_1, name_2):

    """
    This function operates in a similar way to plot_2_cm function - plot and
    compare 2 reports from sklearn's classification_report

    Parameters:
    report_1: The first classification report
    report_2: The second classification report
    name_1 (str): Name for the report_1 on the plot
    name_2 (str): Name for the report_2 on the plot
    """

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

    plt.show()

def plot_model_results(model_1, model_2, model_name,
                       X_test_1, y_test_1,
                       X_test_2, y_test_2, class_names):

    """
    This function compares 2 models: gets test predictions and uses previous
    functions plot_2_reports and plot_2_cm to plot differents between 2 algorithms

    Parameters:
    model_1: The first sklearn model
    model_2: The second sklearn model
    X_test_1: Test data for the first model
    y_test_1: Labels for the first test data
    X_test_2: Test data for the second model
    y_test_2: Labels for the second test data
    class_names (list): Names for the classes
    """

    # Get prediction and classification report for the first model
    y_pred_1 = model_1.predict(X_test_1)
    report_1 = classification_report(y_test_1, y_pred_1,
                                     target_names=class_names, output_dict=True)

    # Get prediction and calssification report for the second model
    y_pred_2 = model_2.predict(X_test_2)
    report_2 = classification_report(y_test_2,y_pred_2,
                                     target_names=class_names, output_dict=True)

    # Plot 2 reports
    plot_2_reports(report_1=report_1, report_2=report_2,
                  name_1=f'{model_name} with reviews', name_2=f'{model_name} without reviews')

    # Get confusion matrices
    cm1 = confusion_matrix(y_test_1,y_pred_1)
    cm2 = confusion_matrix(y_test_2,y_pred_2)

    # Plot confusion matrices
    plot_2_cm(cm1, cm2, name_1=f'{model_name} with reviews confusion matrix',
              name_2=f'{model_name} without reviews confusion matrix',
              class_names=class_names)

def plot_and_compare(pairs_list, model_name, models_dict, data_dict):

    """
    This function compares 2 models trained on different datasets (usually with
    embeddings and without it) and plot results with classification reports and
    confusion matrices using plot_model_results function

    Parameters:
    pairs_list (list): List that contains tuples with names pair of comparable
    data. Values are the keys to data_dict

    data_dict (dict): Dict with the data
    model_name: Name of the model that was trained on 2 different datasets
    models_dict (dict): Dict with trained models
    """

    current_model = models_dict[model_name]

    for pair in pairs_list:

        key_1, key_2 = pair
        model_1, model_2 = current_model[key_1], current_model[key_2]
        _, X_test_1, _, y_test_1 = data_dict[key_1]
        _, X_test_2, _, y_test_2 = data_dict[key_2]

        plot_model_results(model_1, model_2, model_name.upper(),
                           X_test_1, y_test_1,
                           X_test_2, y_test_2, class_names)

def preprocess_params(params, search_type):

    if search_type=='grid_search':

            for k, v in params.items():
                if 'None' in v:
                    v[v.index('None')] = None

    else:

        for k, v in params.items():
            other_types = []
            copied_v = copy.deepcopy(v)
            # Extract strings and currently save them
            for element in copied_v:
                if (type(element) == type('string_type') or type(element)== type(False)) and element != 'None':
                    other_types.append(element)
                    v.pop(v.index(element))

            # Append correct None
            if 'None' in v:
                other_types.append(None)
                v.pop(v.index('None'))

            # Check if parameters have numbers
            if len(v) > 0:
                # Use randint if type is int else use uniform
                if type(v[0]) == type(1): v = [i for i in range(min(v), max(v) + 1)] + other_types 
                else: v = uniform(min(v), max(v)) # Usually if type float then it is the single type for the parameter
            # Else it is string or just None type
            else: v += other_types
            params[k] = v

    return params                           