
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import GridSearchCV

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Input:
        messages.csv: File path of messages data
        categories.csv: File path of categories data
    Output:
        df: Merged dataset from messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    '''
    Input:
        df: Merged dataset from messages and categories
    Output:
        df: Cleaned dataset
    '''
    categories = df['categories'].str.split(';', expand=True)
    new_header = categories.iloc[0] 
    new_header = new_header.str.split('-')
    categories.columns = [row[0] for row in new_header]
    for column in categories:
        categories[column] = [row[1] for row in categories[column].str.split('-')]
        categories[column] = pd.to_numeric(categories[column])
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df,categories], axis=1)
    return df

def save_data(df, database_filename):
    '''
    Save df into sqlite db
    Input:
        df: cleaned dataset
        database_filename: database name
    Output: 
        A SQLite database
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterMessages', engine, index=False)

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print('Cleaning data...')
        df = clean_data(df)        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)        
        print('Cleaned data saved to database!')    
    else:
        print('Example: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()