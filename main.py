import pandas as pd
import numpy as np
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class ScorePredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and numbers
        text = re.sub(rf'[{re.escape(string.punctuation)}\d+]', '', text)

        # Tokenize and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in word_tokenize(text)
                  if token not in self.stop_words and len(token) > 2]

        return ' '.join(tokens)

    def load_and_preprocess_data(self, filepath, text_column, score_column, num_rows=None):
        """Load and preprocess data"""
        df = pd.read_csv(filepath, nrows=num_rows)
        texts = df[text_column].apply(self.preprocess_text)
        scores = df[score_column]
        return texts, scores

    def train_and_evaluate(self, filepath, text_column, score_column,
                           model_save_path=None, num_rows=None):
        """
        Complete training pipeline with TF-IDF
        """
        # Load and preprocess data
        print(f"Loading {num_rows if num_rows else 'all'} rows...")
        texts, scores = self.load_and_preprocess_data(
            filepath, text_column, score_column, num_rows)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, scores, test_size=0.2, random_state=42)

        # TF-IDF Vectorization
        print("Vectorizing ...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Model training
        print("Training Random Forest...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_vec, y_train)

        # Evaluation
        y_pred = self.model.predict(X_test_vec)
        self._print_evaluation_metrics(y_test, y_pred)

        # Save model
        if model_save_path:
            self._save_model(model_save_path)
            print(f"Model saved to {model_save_path}")

    def _print_evaluation_metrics(self, y_true, y_pred):
        """Print evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R²): {r2:.4f}")

    def _save_model(self, filepath):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'preprocessor': {
                'lemmatizer': self.lemmatizer,
                'stop_words': self.stop_words
            }
        }, filepath)


if __name__ == "__main__":
    trainer = ScorePredictor()

    CONFIG = {
        'data_path': 'Data/Amazon Customer Reviews.csv',
        'text_column': 'Text',
        'score_column': 'Score',
        'save_path': 'model.joblib',
        'num_rows': 30000
    }

    try:
        trainer.train_and_evaluate(
            filepath=CONFIG['data_path'],
            text_column=CONFIG['text_column'],
            score_column=CONFIG['score_column'],
            model_save_path=CONFIG['save_path'],
            num_rows=CONFIG['num_rows']
        )
    except FileNotFoundError:
        print(f"Error: File not found at {CONFIG['data_path']}")
    except Exception as e:
        print(f"Training error: {str(e)}")