# import libraries
import sys
import re
import pandas as pd
import numpy as np

import pickle

import sqlite3
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'omw-1.4'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def load_data(database_filepath):
    '''
    A method to load the data from a sql file into a dataframe and define values needed
    for data training

    Args:
    database_filepath - the filepath of the sql file

    Output:
    X - a list of all messages as strings
    y - the dataframe with its respective values
    '''
    # load data from database
    filenameDb = database_filepath.split("/")[1]
    engine = create_engine('sqlite:///' + filenameDb)
    conn = sqlite3.connect(filenameDb)
    filename = filenameDb.split(".")[0]
    
    #Read the sql database into a dataframe
    df = pd.read_sql('SELECT * FROM ' + filename, conn)
    X = df.message.values
    y = df.drop(["id","message","genre","original","related"], axis=1).values
    
    return X, y, df.columns.tolist()


def tokenize(text):
    '''
    A method to analyze the messages as blocks of text. To split the words and use their root forms.
    Then create a list of "tokens" which consist of the root form of each word and its role in a sentence

    Args:
    text - the sentences

    Outputs:
    clean_tokens - a list consisting of the root form of each word in lower case and its role in a sentence 
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    A method that creates a pipeline for building a model to classify messages.
    It consists of the following:
        - CountVectorizer - creating the text documents into a matrix of token counts
        - TfIdfTransformer - converting the text documents intoa  matrix of tf-idf features
        - RandomForsetClassifier - classifying the dataset using RandomForests
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression()))
    ])

    parameters = {
    'tfidf__use_idf': (True, False),
    'tfidf__smooth_idf': [True, False],
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000),
    'clf__estimator__C': [.009,0.01,.09]
}
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    A method to evaluate the classification model
    '''
    # predict on test data
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred))

def save_model(model, model_filepath):
    '''
    A method to save the model into a pickle file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()