# Advanced ML & DL Spam Classification Lab

## Description

This Jupyter notebook (`Advanced_ML_DL_spam_classification_L4 (1).ipynb`) is a comprehensive lab designed to build and compare Machine Learning (ML) and Deep Learning (DL) models for SMS spam classification. The lab explores the intersection of Natural Language Processing (NLP) and Network Security by utilizing the SMS Spam Collection dataset. Participants will learn to preprocess text data, implement various ML algorithms, and construct a Bidirectional LSTM-based DL model to classify messages as spam or ham (non-spam).

The lab covers:
- Data exploration and preprocessing (including lemmatization and stopword removal).
- Visualization of data insights using plots and word clouds.
- Training and evaluation of multiple ML models.
- Building, training, and evaluating a DL model.
- Model saving and prediction on new messages.

## Table of Contents

- [Description](#description)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Models Covered](#models-covered)
- [Results](#results)
- [Author](#author)
- [License](#license)

## Requirements

- Python 3.x
- Jupyter Notebook
- Libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - nltk
  - wordcloud
  - scikit-learn
  - tensorflow (version 2.4 recommended)

## Installation

1. Clone or download the notebook to your local machine.
2. Install the required libraries using pip:

   ```bash
   pip install numpy pandas matplotlib seaborn nltk wordcloud scikit-learn tensorflow==2.4
   ```

3. Download the NLTK data:

   ```python
   import nltk
   nltk.download("stopwords")
   nltk.download("wordnet")
   nltk.download('punkt')
   ```

4. Download the dataset `spam.csv` from [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset) or the alternative URL provided in the notebook.

## Usage

1. Open the notebook in Jupyter:

   ```bash
   jupyter notebook Advanced_ML_DL_spam_classification_L4\ \(1\).ipynb
   ```

2. Run the cells sequentially to:
   - Import libraries and load the dataset.
   - Perform exploratory data analysis (EDA).
   - Preprocess the text data.
   - Train and evaluate ML models (Part A).
   - Build and train the DL model (Part B).
   - Visualize results and save the model.

3. Customize hyperparameters (e.g., vocabulary size, epochs, dropout level) as indicated in the notebook for experimentation.

4. Use the saved model to make predictions on new messages.

## Dataset

The lab uses the **SMS Spam Collection dataset**, which contains SMS messages labeled as 'ham' (legitimate) or 'spam'. The dataset is a subset of publicly available sources, including:
- Grumbletext website
- NUS SMS Corpus
- Caroline Tag's PhD Thesis
- SMS Spam Corpus v.0.1 Big

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Download**: [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset) or alternative URL in the notebook.
- **Size**: Approximately 5,572 messages after preprocessing.
- **Features**:
  - `feature`: Label ('ham' or 'spam')
  - `message`: Raw SMS text

The dataset is imbalanced, with ham messages outnumbering spam by about 4:1.

## Models Covered

### Part A: Advanced Machine Learning
- **Gaussian Naive Bayes**
- **Multinomial Naive Bayes**
- **Decision Tree Classifier**
- **Logistic Regression**
- **K-Neighbors Classifier**
- **Support Vector Classifier (SVC)**
- **Gradient Boosting Classifier**
- **Bagging Classifier** (with SVC, DTC, KNC as base estimators)

Each model includes training, prediction, classification reports, and confusion matrix visualizations.

### Part B: Advanced Deep Learning
- **Bidirectional LSTM Model**:
  - Embedding layer
  - Bidirectional LSTM
  - Dropout layers
  - Dense layers with sigmoid activation for binary classification

The DL model is trained with Adam optimizer, binary cross-entropy loss, and evaluated on accuracy.

## Results

- **ML Models**: Performance metrics (precision, recall, F1-score) are provided for each model via classification reports and confusion matrices. Models like Multinomial Naive Bayes and SVC typically perform well on this task.
- **DL Model**: Achieves high accuracy (often >95%) on the test set. Training history plots show loss and accuracy trends. The model is saved in HDF5 format for future use.
- Visualizations include word frequency plots, word clouds for spam/ham, and confusion matrices.

Example output:
- Test accuracy for DL model: ~97% (varies with hyperparameters).
- Confusion matrices highlight true positives, false positives, etc.

## Author

[JJ](https://www.linkedin.com/in/jonathan-magdy-170324280/))


