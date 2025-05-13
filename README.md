# 📊 Amazon Review Score Predictor

A machine learning model that predicts product review scores (1-5⭐) based on text content.

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 🔠 Text Processing | Tokenization, lemmatization, stopword removal |
| 📊 Vectorization | TF-IDF with 5000 features |
| 🌳 Model | Random Forest Regressor |
| 📈 Evaluation | MSE, RMSE, R² metrics |
| 💾 Persistence | Save/load model functionality |

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/Mohamed-Aboelela-dev/Amazon_Review_Score_Predictor.git
cd sign_language_detection

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
````
Put the "Amazon Customer Reviews.csv" in the "Data" folder.

## 🚀 Quick Start

#### Train the model:
```bash
python main.py
```

#### Test the model:
```bash
python test.py
```

## 📂 Project Structure
```
.
├── main.py             # 🏋️ Training script
├── test.py             # 🧪 Testing script
├── requirements.txt    # 📦 Dependencies
├── model.joblib        # 💾 Saved model (created after training)
└── README.md           # 📖 This file
```

## 📊 Sample Output

Test Results:
Text                                  Expected  Predicted  Error
This product is perfect in every way        5        4.8     0.2
Absolutely terrible quality                 1        1.2     0.2
Mediocre but works okay                     3        2.9     0.1

Average Error: 0.17

## 🛠️ Configuration
Edit these parameters in main.py:
```
CONFIG = {
    'data_path': 'Amazon Customer Reviews.csv',  # 📂 Your dataset
    'text_column': 'Text',                      # 🔤 Text column name
    'score_column': 'Score',                    # ⭐ Rating column name
    'save_path': 'model.joblib',                # 💾 Model save path
    'num_rows': 200                             # 🔢 Rows to use (None for all)
}
```
