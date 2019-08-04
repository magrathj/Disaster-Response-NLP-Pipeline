# Disaster Response Pipeline Project

## Disaster Response Pipeline
This project was completed as part of the course requirements of Udacity's Data Scientist Nanodegree certification.

## Overview
The project used a data set from Figure Eight that contained labeled disaster messages received by an aid organization. A multi-output Random Forrest classifier was trained using supervised learning with a natural language processing (NLP).

An ETL pipeline was created, extracting data from csv files, cleaning and loading into an SQL database. A machine learning pipeline was created to extract the NLP features and then optimize the algorithm using grid search. A web app was then developed that extracts the initial data from the database and provides some interactive visual summaries. Users are also able to enter their own message to be classified by the algorithm.

## Technologies Used
Python
Libraries: pandas, sklearn, sqlite3, sqlalchemy, nltk, plotly, flask
HTML
Bootstrap
Project Details
Web App
Users have the ability to enter their own message to be classified.

'Enter Message'

They will then be shown the classification results.

'Example Classification'

They can also see a summary of the original dataset. 'Message Types' 'Related Types' 'Message Correlation'

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
