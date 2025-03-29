import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
# import sys

# sys.path.append("../dataloader")
# from dataloader import daigtv2_loader
# sys.path.append("../part1")

# data = daigtv2_loader("~/Desktop/ISyE4600_Machine_Learning/ISyE4600 Project/")

import sys
import os

# Get absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add dataloader directory to sys.path
dataloader_path = os.path.join(project_root, "dataloader")
sys.path.append(dataloader_path)

# Now import the module
from dataloader import daigtv2_loader

# Load the data
data = daigtv2_loader("~/Desktop/ISyE4600_Machine_Learning/ISyE4600 Project/")

class AITextClassifier:
    """
    This class handles training two classical ML models (SVM and Random Forest) 
    to differentiate AI-generated writing from human writing. Although it can 
    leverage all columns for filtering or other data processing steps during 
    training, the final prediction uses only the text column.
    """
    def __init__(self):
        # Define the vectorizer and models
        self.vectorizer = TfidfVectorizer(
            token_pattern=r'\b\w+\b',
            lowercase=True
        )
        self.svm_model = SVC(kernel='linear', random_state=42)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the vectorizer and both models on the provided training DataFrame.
        
        Parameters:
        - df: DataFrame containing 'text' and 'label' columns, plus any others.
        
        This method could optionally filter or process the data using the 
        additional columns if needed. For example, you might remove certain rows 
        based on 'prompt_name', 'source', or 'RDizzl3_seven'. Here, it's kept simple.
        """
        # Example: potential filtering or data preparation using extra columns
        # df = df[df["source"] != "some_source_to_exclude"]  # if you had such a condition

        X = df["text"]
        y = df["label"]

        # Transform text to TF-IDF features
        X_tfidf = self.vectorizer.fit_transform(X)

        # Train the SVM model
        self.svm_model.fit(X_tfidf, y)

        # Train the Random Forest model
        self.rf_model.fit(X_tfidf, y)

        self.is_fitted = True

    def predict_svm(self, texts) -> pd.Series:
        """
        Predict labels for a list of texts using the trained SVM model.
        Only the text is used for the prediction.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        X_tfidf = self.vectorizer.transform(texts)
        return self.svm_model.predict(X_tfidf)

    def predict_rf(self, texts) -> pd.Series:
        """
        Predict labels for a list of texts using the trained Random Forest model.
        Only the text is used for the prediction.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        X_tfidf = self.vectorizer.transform(texts)
        return self.rf_model.predict(X_tfidf)

    def evaluate(self, df: pd.DataFrame) -> None:
        """
        Evaluate both models on a DataFrame containing 'text' and 'label'.
        Predictions are made using text only, but the method compares 
        the predictions with the actual labels in df['label'].
        """
        X = df["text"]
        y_true = df["label"]

        # Predictions
        svm_preds = self.predict_svm(X)
        rf_preds = self.predict_rf(X)

        # Evaluation metrics
        print("=== SVM Results ===")
        print("Accuracy:", accuracy_score(y_true, svm_preds))
        print(classification_report(y_true, svm_preds))

        print("=== Random Forest Results ===")
        print("Accuracy:", accuracy_score(y_true, rf_preds))
        print(classification_report(y_true, rf_preds))


# Example usage
if __name__ == "__main__":
    # Suppose 'data' is your complete DataFrame with columns:
    # ['text', 'label', 'prompt_name', 'source', 'RDizzl3_seven']
    #data = pd.read_csv("your_dataset.csv")

    # Split the data
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    # Initialize the classifier
    classifier = AITextClassifier()
    
    # Train the classifier on the training data
    classifier.fit(train_df)
    
    # Evaluate on the test data (prediction is based on text only)
    classifier.evaluate(test_df)
    #pass
