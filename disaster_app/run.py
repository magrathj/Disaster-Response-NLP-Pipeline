import json
import plotly
import pandas as pd

import nltk
print("hey")

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sqlalchemy.engine import reflection
#from functions import tokenize

app = Flask(__name__)



# load data
print("*******************************")
engine = create_engine('sqlite:///disaster_app/DisasterResponse.db')
insp = reflection.Inspector.from_engine(engine)
print(insp.get_table_names())
print("********engine************")
df = pd.read_sql_table('DisasterMessages', engine)
print("*****************************")
# load model
#model = joblib.load("disaster_app/classifier.pkl")
print("model")

# index webpage displays cool visuals and receives user input text for model
@app.route('/', methods=['POST', 'GET'])
@app.route('/index', methods=['POST', 'GET'])
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

