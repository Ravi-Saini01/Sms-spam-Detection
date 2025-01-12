# SMS Spam Detection System

## Overview

The **SMS Spam Detection System** is a machine learning project aimed at classifying SMS messages as either spam or ham (not spam). By leveraging natural language processing (NLP) techniques and machine learning algorithms, the project preprocesses and analyzes SMS text to identify spam messages effectively.

---

## Features

- **Data Cleaning:** Handles missing values, removes duplicates, and renames columns for clarity.
- **Exploratory Data Analysis (EDA):** Includes data visualization (e.g., pie charts, histograms, and descriptive statistics) to understand the dataset.
- **Text Preprocessing:** Implements tokenization, stopword removal, stemming, and punctuation handling for efficient text analysis.
- **Visualization:** Generates word clouds for spam and ham messages.
- **Model Building:** Trains multiple Naive Bayes models (Gaussian, Multinomial, and Bernoulli).
- **Evaluation Metrics:** Calculates accuracy, confusion matrix, and precision scores to assess model performance.
- **Real-time Prediction:** Allows input of custom SMS messages to predict whether they are spam or ham.

---

## Dataset

The project uses the **spam.csv** dataset, which contains SMS messages labeled as:

- **ham:** Legitimate messages.
- **spam:** Unsolicited promotional or fraudulent messages.

### Dataset Columns:

1. **v1**: The label (ham/spam).
2. **v2**: The text content of the SMS message.

---

## Installation

### Prerequisites

Ensure you have Python 3.x installed on your system.

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sms-spam-detection.git
   cd sms-spam-detection
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the NLTK datasets:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

---

## Usage

### Training the Model

Run the main script to preprocess the data, train the model, and evaluate its performance:

```bash
python main.py
```

### Predicting Spam

Use the trained model to classify a new SMS message:

```python
from joblib import load
model = load('models/trained_model.joblib')

tfidf = load('models/tfidf_vectorizer.joblib')

sample_message = "Congratulations! You've won a $1000 gift card. Reply WIN to claim."
sample_vectorized = tfidf.transform([sample_message])
prediction = model.predict(sample_vectorized)
print("Spam" if prediction[0] == 1 else "Not Spam")
```

---

## Results

The **Multinomial Naive Bayes** model performed best, achieving:

- **Accuracy:** High predictive accuracy on test data.
- **Precision:** Minimal false positives for spam classification.

---

## File Structure

```
sms-spam-detection/
|
├── spam.csv                # Dataset
├── main.py                 # Main script for training and prediction
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
└── models/
    ├── trained_model.joblib   # Trained model
    └── tfidf_vectorizer.joblib # TF-IDF Vectorizer
```

---

## Dependencies

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Seaborn
- Matplotlib
- WordCloud

Install these dependencies using:

```bash
pip install -r requirements.txt
```

---

## Visualizations

1. **Word Clouds:**
   - Spam and ham messages visualized using word clouds for pattern recognition.
2. **Histograms:**
   - Distribution of message length, word count, and sentence count.

---

## Future Work

- Implement deep learning techniques for improved performance.
- Expand the dataset to include multilingual SMS messages.
- Deploy the system as a web or mobile application for real-world use.

---



## Acknowledgments

Special thanks to the contributors of the spam dataset and the open-source libraries that made this project possible.

---

##
