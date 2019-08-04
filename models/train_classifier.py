import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])


import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from scipy.stats import hmean
from scipy.stats.mstats import gmean
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import sys
from sqlalchemy import create_engine
import pickle
from sqlalchemy import create_engine

def load_data(database_filepath):
    '''
    Load data from database as dataframe
    Input:
        database.csv: File path of sql database
    Output:
        X: Message data (features)
        Y: Categories (target)
        category_names: Labels for 36 categories
    '''
    print('load_data')
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterMessages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return X, Y, category_names


def tokenize(text):
    '''
    Tokenize and clean text
    Input:
        text: original message text
    Output:
        clean_tokens: Tokenized, cleaned, and lemmatized text
    '''
    print("tokenize")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)



def build_model():
    """
    Build Model function
    
    This function output is a Scikit ML Pipeline that process text messages
    according to NLP best-practice and apply a classifier.
    """
    model = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance using test data
    Input: 
        model: Model to be evaluated
        X_test: Test data (features)
        Y_test: True lables for Test data
        category_names: Labels for 36 categories
    Output:
        Print accuracy and classfication report for each category
    '''
    print("evaluate_model")
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))


def save_model(model, model_filepath):
    '''
    Save model as a pickle file 
    Input: 
        model: Model to be saved
        model_filepath: path of the output pick file
    Output:
        A pickle file of saved model
    '''
    print("save_model")
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        model = build_model()
        model.fit(X_train, Y_train)
        evaluate_model(model, X_test, Y_test, category_names)
        save_model(model, model_filepath)

    else:
        print('Example: python train_classifier.py ../data/DisasterResponse.db disaster_app/classifier.pkl')


if __name__ == '__main__':
    main()