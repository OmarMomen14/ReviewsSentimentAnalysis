# Designing Sentiment Analysis Tool For Women’s Fashion e-Commerce

A flask application that analyzes customers reviews to classify it in two categories, “Positive Reviews” or “Negative Reviews”.

## Application Architecture

### 1. Receipt of User Reviews
- System receives reviews of products from the database of the e-commerce website as input.
- This input is then sent for data preprocessing.

### 2. Data Preprocessing
- Remove URL characters such as http://, https://
- Tokenize words
- Normalize words
  - Remove non-ASCII characters
  - Change to lower-case
  - Remove punctuation
  - Remove stopwords
- Lemmatize
- Encode sentence
- Pad sequences

### 3. Prediction of Review Polarity
- Input the preprocessed data into a sequential neural network with 5 layers
  - Embedding
  - Bidirectional
  - Dropout
  - Fully connected
  - Fully connected
- Neural network predicts the polarity of the input reviews.
- Predictions are sent to the e-commerce website database for data analysis
