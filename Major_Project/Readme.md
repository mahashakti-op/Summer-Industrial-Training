# AI Misinformation Detector using Machine Learning

## Project Description

This beginner-friendly college project classifies a news article or social media post as **REAL** (`0`) or **FAKE** (`1`). It uses traditional Natural Language Processing and Machine Learning: cleaned title and article text are converted into TF-IDF features, then three classifiers are compared automatically.

The accompanying research paper inspired the topic and the transparent classical-ML pipeline. This repository intentionally uses only the requested traditional models, rather than deep learning or LLMs.

## Features

- Complete EDA in a single Jupyter notebook
- Reusable text cleaning with lowercasing, URL/HTML/number/punctuation removal, stopword removal, lemmatization, and whitespace normalization
- Text-length and word-count visualizations, class distribution, word clouds, and frequent-word analysis
- TF-IDF features with 5,000 maximum features and unigrams/bigrams
- Comparison of Logistic Regression, Multinomial Naive Bayes, and Linear SVM
- Accuracy, precision, recall, F1-score, classification reports, and confusion matrices
- Automatic F1-score-based best-model selection
- Streamlit interface for interactive predictions

## Dataset

Place the dataset at `dataset/fake_news.csv`. It must contain these columns:

| Column | Meaning |
| --- | --- |
| `title` | News headline or post title |
| `text` | Main article or social media text |
| `label` | `0` for real and `1` for fake |

A small demonstration dataset is included so the notebook can run immediately. Replace it with a larger, properly labelled dataset before using the project results in a report or presentation.

## Models Used

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine (Linear SVM)

The notebook selects and saves the model with the highest test-set F1-score for the fake-news class.

## Installation

```bash
cd AI_Misinformation_Detector
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## How to Run

1. Open `notebook.ipynb` in Jupyter Notebook or VS Code.
2. Run every cell from top to bottom. The notebook downloads the required small NLTK language resources when necessary.
3. After training, `model.pkl` and `tfidf_vectorizer.pkl` will be created in the project folder.
4. Start the application:

```bash
streamlit run app.py
```

5. Enter a title and/or article text, then select **Analyze text**.

## Future Improvements

- Train on a larger, balanced and regularly updated dataset
- Add cross-validation and hyperparameter tuning
- Display the most influential TF-IDF words for a prediction
- Add source, author, publication date, and claim-verification features
- Evaluate fairness and performance across topics, languages, and writing styles
- Add human fact-checking and source links before any real-world decision

## Important Note

This project learns patterns in its training data. It does not verify factual truth, and its predictions should never be treated as a final fact-checking decision.
