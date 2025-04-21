# ISYE4600_Final_Project
NLP Overview: AI text detection

You first need to download the data from the following link and put it in the parent directly of the cloned directory.
 https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset/data?select=train_v2_drcat_02.csv

Also, make sure all the required libraries in requirements.txt are downloaded to your local environment.
 
dataloader: loads the data from Kaggle.

Part 1: training SVM and Random Forest models with TF-IDF vectorization.
- estimated running time: 15m

Part 2: training LSTM with FastText vectorization. Download wiki-news-300d-1M.vec from this link: https://fasttext.cc/docs/en/english-vectors.html and put it in the parent directly of the cloned directory to run part2 from scratch.
- p2.ipynb estimated running time of p2.ipynb: 30m with GPU usage
- p2.ipynb trains the model from scratch.
- To load the pretrained model weights and test new phrases, edit and run part2.py. FYI, the model cannot make effective predictions for phrases that are too short.
  
Part 3: training BERT(Transformer) with Catboost.
- part3.ipynb estimated running time: 1h with GPU usage
- part3.ipynb trains the model from scratch
