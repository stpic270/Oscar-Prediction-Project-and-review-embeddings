#!/bin/bash
python src/preprocess_text_embeddings.py -cat bestpicture -pr -sl && python src/preprocess_dataframe.py -cat bestpicture -us -rs 43 -sl && python src/preprocess_dataframe.py -cat bestpicture -us -we -rs 43 -sl

python src/data_analysis.py -us True -cat bestpicture -tn 60 

python src/param_searcher.py -m log_reg -st grid_search -cat bestpicture -us -v 1 -sp -sl && python src/train.py -m log_reg -cat bestpicture -us -sl -pr && python src/test.py -m log_reg -cat bestpicture -sl -sg 
python src/param_searcher.py -m log_reg -st grid_search -cat bestpicture -us -we -v 1 -sp -sl && python src/train.py -m log_reg -cat bestpicture -us -we -sl -pr && python src/test.py -m log_reg -cat bestpicture -sl -sg 

python src/param_searcher.py -m rf -st grid_search -cat bestpicture -us -v 1 -sp -sl && python src/train.py -m rf -cat bestpicture -us -sl -pr && python src/test.py -m rf -cat bestpicture -sl -sg 
python src/param_searcher.py -m rf -st grid_search -cat bestpicture -us -we -v 1 -sp -sl && python src/train.py -m rf -cat bestpicture -us -we -sl -pr && python src/test.py -m rf -cat bestpicture -sl -sg 

python src/param_searcher.py -m svm -st grid_search -cat bestpicture -us -v 1 -sp -sl && python src/train.py -m svm -cat bestpicture -us -sl -pr && python src/test.py -m svm -cat bestpicture -sl -sg 
python src/param_searcher.py -m svm -st grid_search -cat bestpicture -us -we -v 1 -sp -sl && python src/train.py -m svm -cat bestpicture -us -we -sl -pr && python src/test.py -m svm -cat bestpicture -sl -sg 

python src/model_compare.py -us True -cat bestpicture -sl

sleep infinity