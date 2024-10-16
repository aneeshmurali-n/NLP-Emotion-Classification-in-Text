# NLP Emotion Classification in Text

This project focuses on classifying emotions (fear, anger, joy) in text comments using Natural Language Processing (NLP) techniques and machine learning models.

## Project Overview

The project involves the following steps:

1. **Data Loading and Preprocessing:** Loads a dataset of comments and emotions, cleans the text by removing unnecessary characters and converting to lowercase, and tokenizes the text into individual words.
2. **Feature Extraction:** Creates numerical representations of the text using the Bag-of-Words (BoW) and TF-IDF methods.
3. **Model Development:** Trains two machine learning models - Naive Bayes and Support Vector Machine (SVM) - using the extracted features and emotion labels.
4. **Model Evaluation and Comparison:** Evaluates the performance of both models using metrics like accuracy, precision, recall, and F1-score, and compares their results.

## Results

The SVM classifier with BoW achieved the highest accuracy (91.7%) and demonstrated superior performance across various evaluation metrics. It is identified as the most suitable model for emotion classification in this project.

| Model | Accuracy | Precision (macro avg) | Recall (macro avg) | F1-score (macro avg) |
|---|---|---|---|---|
| SVM classifier with bow | 0.9166 | 0.92 | 0.92 | 0.92 |
| SVM classifier with TF-IDF | 0.9141 | 0.92 | 0.91 | 0.91 |
| Naive Bayes classifier with bow | 0.8905 | 0.89 | 0.89 | 0.89 |
| Naive Bayes classifier with TF-IDF | 0.8947 | 0.90 | 0.89 | 0.89 |

## Usage

To run this project:

1. Clone the repository.
2. Install the required libraries: `nltk`, `pandas`, `scikit-learn`.
3. Execute the Jupyter Notebook provided in the repository.
or [Open Project in Google Colab](https://colab.research.google.com/github/aneeshmurali-n/NLP-Emotion-Classification-in-Text/blob/631d66678a0b161a113698cc93b2800370792570/NLP_Emotion_Classification_in_Text.ipynb)

## Dataset

The dataset used in this project is located in this repository:  `nlp_dataset.csv`.

## License

This project is licensed under the MIT License.
