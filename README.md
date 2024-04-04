# Oscar-Prediction-Project-and-review-embeddings
This project offers you to train models to predict oscar winners using additional text emeddings in the following categories: Best Picture, Best Director, Best Actor, Best Actress, Best Supporting Actor, Best Supporting Actress

This project has 3 models: SVM, random forest and logistic regression. It laso contains 3 dataframes that have data and oscar winner labels for actors, director and picture movies. You could train your model with the following guide
First of all you need to choose usage type (of course you could move from one to another during usage)

## Begginer 

1) All types start with preprocessing embeddings (Use src/preprocess_text_embeddings.py --help to get description for each parameter) 

#### - python src/preprocess_text_embeddings.py -cat bestpicture -pr -sl

2) After that you need to choose what data to append to the main dataframe and then scale it (Nothing, SMOTE or text embeddings or both. Use src/preprocess_dataframe.py --help to get description for parameters). HINT always preprocess data with random_state (-rs flag):

#### - python src/preprocess_dataframe.py -cat bestpicture -us -rs 43 -sl

3) Finally, you can train train your model (check src/model_params/main_params.yaml and try different model parameters. Use python src/train.py --help to get description for parameters)

#### - python src/train.py -m log_reg -cat bestpicture -us -sl -pr 

4) Run test and see results

#### - python src/test.py -m log_reg -cat bestpicture -sl -sg 

## Intermediate  

This level has additional actions to the begginer level

Find the best parameters manually is a bit hard, so before the third step try this (Use python src/param_searcher.py --help to get description for parameters): 

#### - python src/param_searcher.py -m log_reg -st grid_search -cat bestpicture -us -v 3 -sp -sl

This function will print the best parameters but you also could specify to load them to src/model_params/main_params.yaml by -sp flag

## Advanced

Besides model settings, it is also important to check your data so before the second stage from the begginer level check data distribution with TSNE and preferable cluster number with the following command (Use python src/data_analysis.py --help to get parameters description)

#### - python src/data_analysis.py -us True -cat bestpicture -tn 60 

## You also can get reviews for your movies, just check notebooks/Project_add_more_data.ipynb 

## So, try improve models prediction ability for the Best Picture category or try this repo to get results for the 2 other datasets oscardata_acting.csv and oscardata_bestdirector.csv

# Coming soon

- Ability to add parse data by movie names
- Applying reviews parsing to the data from notebooks/Project_add_more_data.ipynb notebook
- Utilizeng different ways of parsing reviews
- Using more number types of language models for text embeddings extracting

# References

The author of this repo was inspired by this repo https://github.com/MateVaradi/OscarPrediction. Datasets were also extracted from mentioned repo 

