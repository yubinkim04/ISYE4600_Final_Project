# Imports for basic processing and system handling
import os
import sys
import pandas as pd
import numpy as np

# In Jupyter, __file__ is not available; use os.getcwd() or specify the path manually.
sys.path.append("../dataloader")
from dataloader import daigtv2_loader
sys.path.append("../part1")
# Now import your custom loader module (make sure the 'dataloader' folder is in your project root)
from dataloader import daigtv2_loader


# Set the path to your dataset folder
data_path = "~/Desktop/ISyE4600_Machine_Learning/ISyE4600_Project/"

# Load the data using the custom data loader
data = daigtv2_loader(data_path)

# Optionally, inspect the data
print("Data shape:", data.shape)
print(data.head())


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

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
            # token_pattern=r'\b\w+\b',
            lowercase=True
        )
        self.svm_model = SVC(kernel='linear', random_state=42)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the vectorizer and both models on the provided training DataFrame.
        The DataFrame should contain 'text' and 'label' columns.
        """
        X = df["text"]
        y = df["label"]

        # Transform text to TF–IDF features
        X_tfidf = self.vectorizer.fit_transform(X)

        # Train the SVM model
        self.svm_model.fit(X_tfidf, y)

        # Train the Random Forest model
        self.rf_model.fit(X_tfidf, y)

        self.is_fitted = True

    def predict_svm(self, texts) -> pd.Series:
        """
        Predict labels for a list of texts using the trained SVM model.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        X_tfidf = self.vectorizer.transform(texts)
        return self.svm_model.predict(X_tfidf)

    def predict_rf(self, texts) -> pd.Series:
        """
        Predict labels for a list of texts using the trained Random Forest model.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        X_tfidf = self.vectorizer.transform(texts)
        return self.rf_model.predict(X_tfidf)

    def evaluate(self, df: pd.DataFrame) -> None:
        """
        Evaluate both models on a DataFrame containing 'text' and 'label'.
        Predictions are made using the text only, then compared to the actual labels.
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

# Split the data: use 20% to 40% of the data as test data
total_len = len(data)
start = int(total_len * 0.2)
end = int(total_len * 0.4)
test = data.iloc[start:end]
train = pd.concat([data.iloc[:start], data.iloc[end:]])

# Initialize the classifier and train on the training data
classifier = AITextClassifier()
classifier.fit(train)

# Evaluate both the SVM and Random Forest models on the test data
classifier.evaluate(test)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

class TreeVisualizer:
    def __init__(self, classifier):
        self.classifier = classifier
        self.model = classifier.rf_model
        self.vectorizer = classifier.vectorizer

    def plot_tree_from_forest(self, tree_index=0, class_names=None, max_features=50):
        """
        Visualize one decision tree from the random forest.

        Parameters:
        - tree_index: Index of the tree in the forest to visualize.
        - class_names: List of class names (e.g., ["Spam", "Not Spam"]).
        - max_features: Limit the number of feature names to plot for readability.
        """
        if tree_index >= len(self.model.estimators_):
            raise IndexError(f"Tree index {tree_index} out of range.")

        estimator = self.model.estimators_[tree_index]
        feature_names = self.vectorizer.get_feature_names_out()

        # if max_features and len(feature_names) > max_features:
        #     print(f"⚠️ Feature space is large ({len(feature_names)}). Showing only the first {max_features} features.")
        #     feature_names = feature_names[:max_features]
        feature_names = self.vectorizer.get_feature_names_out()

        plt.figure(figsize=(20, 10))
        plot_tree(
            estimator,
            filled=True,
            feature_names=feature_names,
            class_names=class_names or ["Human", "AI"],
            fontsize=8
        )
        plt.title(f"Random Forest Decision Tree Visualization (Estimator {tree_index})")
        plt.show()

visualizer = TreeVisualizer(classifier)
visualizer.plot_tree_from_forest(tree_index=0, class_names=["Human", "AI"])

# with a better visualization

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree

class TFIDFTreeVisualizer:
    def __init__(self, max_features=1000, max_depth=3, min_samples_leaf=10):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self.model = DecisionTreeClassifier(
            random_state=42,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf
        )
        self.feature_names = None

    def fit(self, texts, labels):
        """
        Fit the TF-IDF vectorizer and decision tree model.
        """
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.feature_names = self.vectorizer.get_feature_names_out()

    def plot(self, class_names=["Class 0", "Class 1"], figsize=(18, 8), title="CART Tree Visualization"):
        """
        Plot the trained decision tree with readable layout.
        """
        if self.feature_names is None:
            raise ValueError("Model has not been fitted. Call `fit()` first.")

        plt.figure(figsize=figsize)
        plot_tree(
            self.model,
            filled=True,
            feature_names=self.feature_names,
            class_names=class_names,
            max_depth=self.model.get_depth(),
            fontsize=10,
            impurity=False,
            proportion=True
        )
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    # Example usage
    visualizer = TFIDFTreeVisualizer(
        max_features=1000,
        max_depth=3,
        min_samples_leaf=10
    )

    # Fit with training data
    visualizer.fit(train["text"], train["label"])

    # Plot the tree
    visualizer.plot(class_names=["Human written", "AI written"])

