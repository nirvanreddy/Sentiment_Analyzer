from django.http import HttpResponse
import json
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.shortcuts import render
import csv
from django.views import View
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.corpus import stopwords
import re
import nltk
import warnings
import twint
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence
classifier = TextClassifier.load('en-sentiment')
nltk.download('stopwords')
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


warnings.filterwarnings('ignore')


def home(request):
    return render(request, "home.html")

# Cleaning the text
def cleanTXT(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'RT[/s]', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = ' '.join(word for word in text.split()
                    if word not in stopwords.words('english'))
    return text

# Create a function to get the subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# creating a function to get the polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# Creating a func. to compute negative,positive and neutral
def getAnalysis(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"


def search_tweet(search_item):
    c = twint.Config()
    c.Search = search_item  # the search terms you want to scrape
    c.Limit = 200  # the number of tweets to scrape
    c.Lang = "en"  # the language of the tweets
    c.Pandas = True  # store the scraped tweets in a Pandas dataframe
    twint.run.Search(c)
    return twint.storage.panda.Tweets_df


def display_tweets(text_df):
    tweets = text_df.head(15).to_records(index=False)
    return tweets


def get_vader_sentiment(text):
    sentiment = sia.polarity_scores(text)
    compound_score = sentiment['compound']

    if compound_score > 0:
        return 'Positive'
    elif compound_score < 0:
        return 'Negative'
    else:
        return 'Neutral'
    

    
def get_flair_sentiment(text):
    # create a sentence object
    sentence = Sentence(text)

    # predict the sentiment of the sentence
    classifier.predict(sentence)

    # get the predicted label
    label = sentence.labels[0].value
    return label

def analyse_ht(request):
    if request.method == 'POST':
        ht = request.POST.get('Hashtag')
        global tweets_df
        tweets_df = search_tweet(ht)
        tweets_df['tweet'] = tweets_df['tweet'].apply(cleanTXT)
        
        #Using TextBlob Library
        tweets_df['T_Subjectivity'] = tweets_df['tweet'].apply(getSubjectivity)
        tweets_df['T_Polarity'] = tweets_df['tweet'].apply(getPolarity)
        tweets_df['T_Analysis'] = tweets_df['T_Polarity'].apply(getAnalysis)

        # Calculating Accuracy for TextBlob
        sample = tweets_df.sample(n=100, random_state=42)
        train, valid = train_test_split(sample, test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        X_train = vectorizer.fit_transform(train['tweet'])
        y_train = train['T_Analysis']
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        X_valid = vectorizer.transform(valid['tweet'])
        y_valid = valid['T_Analysis']
        y_pred = clf.predict(X_valid)
        T_accuracy = accuracy_score(y_valid, y_pred)

        # Using VADER Library
        tweets_df['S_Analysis'] = tweets_df['tweet'].apply(get_vader_sentiment)

        # Calculating Accuracy for VADER
        sample = tweets_df.sample(n=100, random_state=42)
        train, valid = train_test_split(sample, test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        X_train = vectorizer.fit_transform(train['tweet'])
        y_train = train['S_Analysis']
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        X_valid = vectorizer.transform(valid['tweet'])
        y_valid = valid['S_Analysis']
        y_pred = clf.predict(X_valid)
        S_accuracy = accuracy_score(y_valid, y_pred)

        #Using Flair Library
        tweets_df['F_Analysis'] = tweets_df['tweet'].apply(get_vader_sentiment)

        # Calculating Accuracy for Flair Library
        sample = tweets_df.sample(n=100, random_state=42)
        train, valid = train_test_split(sample, test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        X_train = vectorizer.fit_transform(train['tweet'])
        y_train = train['F_Analysis']
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        X_valid = vectorizer.transform(valid['tweet'])
        y_valid = valid['F_Analysis']
        y_pred = clf.predict(X_valid)
        F_accuracy = accuracy_score(y_valid, y_pred)

        max_accuracy = max(F_accuracy,S_accuracy,T_accuracy)


        if (max_accuracy == T_accuracy):
            ptweets = tweets_df[tweets_df.T_Analysis == 'Positive']
            pt1 = round((ptweets.shape[0]/tweets_df.shape[0]) * 100, 1)

            ntweets = tweets_df[tweets_df.T_Analysis == 'Negative']
            nt1 = round((ntweets.shape[0]/tweets_df.shape[0]) * 100, 1)

            neutweets = tweets_df[tweets_df.T_Analysis == 'Neutral']
            neut1 = round((neutweets.shape[0]/tweets_df.shape[0]) * 100, 1)

            tweets_df['Analysis']=tweets_df['T_Analysis']
            dsp_tweets = display_tweets(tweets_df)
            ans = "Textblob"


            context1 = {
                'ans': ans,
                'dsp_tweets': dsp_tweets,
                'ht': ht,
                'pt1': pt1,
                'nt1': nt1,
                'neut1': neut1,
                'accuracy': T_accuracy
            }
        elif(max_accuracy == S_accuracy):
            ptweets = tweets_df[tweets_df.S_Analysis == 'Positive']
            pt1 = round((ptweets.shape[0]/tweets_df.shape[0]) * 100, 1)

            ntweets = tweets_df[tweets_df.S_Analysis == 'Negative']
            nt1 = round((ntweets.shape[0]/tweets_df.shape[0]) * 100, 1)

            neutweets = tweets_df[tweets_df.S_Analysis == 'Neutral']
            neut1 = round((neutweets.shape[0]/tweets_df.shape[0]) * 100, 1)
            tweets_df['Analysis']=tweets_df['S_Analysis']
            dsp_tweets = display_tweets(tweets_df)
            ans = "Vader"

            context1 = {
                'ans': ans,
                'dsp_tweets': dsp_tweets,
                'ht': ht,
                'pt1': pt1,
                'nt1': nt1,
                'neut1': neut1,
                'accuracy': S_accuracy
            }
        elif(max_accuracy == F_accuracy):
            ptweets = tweets_df[tweets_df.S_Analysis == 'POSITIVE']
            pt1 = round((ptweets.shape[0]/tweets_df.shape[0]) * 100, 1)

            ntweets = tweets_df[tweets_df.S_Analysis == 'NEGATIVE']
            nt1 = round((ntweets.shape[0]/tweets_df.shape[0]) * 100, 1)

            neutweets = tweets_df[tweets_df.S_Analysis == 'NEUTRAL']
            neut1 = round((neutweets.shape[0]/tweets_df.shape[0]) * 100, 1)
            tweets_df['Analysis']=tweets_df['F_Analysis']
            dsp_tweets = display_tweets(tweets_df)
            ans = "Flair"

            context1 = {
                'ans': ans,
                'dsp_tweets': dsp_tweets,
                'ht': ht,
                'pt1': pt1,
                'nt1': nt1,
                'neut1': neut1,
                'accuracy': F_accuracy
            }

        return render(request, "output.html", context1)


def upload_form(request):
    if request.method == 'POST':
        file = request.FILES['csv_file']
        df = pd.read_csv(file)

        text_df = df.drop(['id'], axis=1)
        text_df['tweet'] = text_df['tweet'].apply(cleanTXT)

        text_df['T_Subjectivity'] = text_df['tweet'].apply(getSubjectivity)
        text_df['T_Polarity'] = text_df['tweet'].apply(getPolarity)
        text_df['T_Analysis'] = text_df['T_Polarity'].apply(getAnalysis)

        # Calculating Accuracy for TextBlob
        sample = text_df.sample(n=100, random_state=42)
        train, valid = train_test_split(sample, test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        X_train = vectorizer.fit_transform(train['tweet'])
        y_train = train['T_Analysis']
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        X_valid = vectorizer.transform(valid['tweet'])
        y_valid = valid['T_Analysis']
        y_pred = clf.predict(X_valid)
        T_accuracy = accuracy_score(y_valid, y_pred)

        # Using Vader Library
        text_df['S_Analysis'] = text_df['tweet'].apply(get_vader_sentiment)

        # Calculating Accuracy for Vader
        sample = text_df.sample(n=100, random_state=42)
        train, valid = train_test_split(sample, test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        X_train = vectorizer.fit_transform(train['tweet'])
        y_train = train['S_Analysis']
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        X_valid = vectorizer.transform(valid['tweet'])
        y_valid = valid['S_Analysis']
        y_pred = clf.predict(X_valid)
        S_accuracy = accuracy_score(y_valid, y_pred)

         #Using Flair Library
        text_df['F_Analysis'] = text_df['tweet'].apply(get_vader_sentiment)

        # Calculating Accuracy for Flair Library
        sample = text_df.sample(n=100, random_state=42)
        train, valid = train_test_split(sample, test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        X_train = vectorizer.fit_transform(train['tweet'])
        y_train = train['F_Analysis']
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        X_valid = vectorizer.transform(valid['tweet'])
        y_valid = valid['F_Analysis']
        y_pred = clf.predict(X_valid)
        F_accuracy = accuracy_score(y_valid, y_pred)

        max_accuracy = max(F_accuracy,S_accuracy,T_accuracy)


        if (max_accuracy == T_accuracy):
            ptweets = text_df[text_df.T_Analysis == 'Positive']
            pt1 = round((ptweets.shape[0]/text_df.shape[0]) * 100, 1)

            ntweets = text_df[text_df.T_Analysis == 'Negative']
            nt1 = round((ntweets.shape[0]/text_df.shape[0]) * 100, 1)

            neutweets = text_df[text_df.T_Analysis == 'Neutral']
            neut1 = round((neutweets.shape[0]/text_df.shape[0]) * 100, 1)
            text_df['Analysis']=text_df['T_Analysis']
            dsp_tweets = display_tweets(text_df)
            ans = "Textblob"

            context1 = {
                'ans': ans,
                'dsp_tweets': dsp_tweets,
                'pt1': pt1,
                'nt1': nt1,
                'neut1': neut1,
                'accuracy': T_accuracy
            }
        elif(max_accuracy == S_accuracy):
            ptweets = text_df[text_df.S_Analysis == 'Positive']
            pt1 = round((ptweets.shape[0]/text_df.shape[0]) * 100, 1)

            ntweets = text_df[text_df.S_Analysis == 'Negative']
            nt1 = round((ntweets.shape[0]/text_df.shape[0]) * 100, 1)

            neutweets = text_df[text_df.S_Analysis == 'Neutral']
            neut1 = round((neutweets.shape[0]/text_df.shape[0]) * 100, 1)
            text_df['Analysis']=text_df['T_Analysis']
            dsp_tweets = display_tweets(text_df)
            ans = "Vader"

            context1 = {
                'ans': ans,
                'dsp_tweets': dsp_tweets,
                'pt1': pt1,
                'nt1': nt1,
                'neut1': neut1,
                'accuracy': S_accuracy
            }
        elif(max_accuracy == F_accuracy):
            ptweets = text_df[text_df.S_Analysis == 'POSITIVE']
            pt1 = round((ptweets.shape[0]/text_df.shape[0]) * 100, 1)

            ntweets = text_df[text_df.S_Analysis == 'NEGATIVE']
            nt1 = round((ntweets.shape[0]/text_df.shape[0]) * 100, 1)

            neutweets = text_df[text_df.S_Analysis == 'NEUTRAL']
            neut1 = round((neutweets.shape[0]/text_df.shape[0]) * 100, 1)
            text_df['Analysis']=text_df['T_Analysis']
            dsp_tweets = display_tweets(text_df)
            ans = "Flair"

            context1 = {
                'ans': ans,
                'dsp_tweets': dsp_tweets,
                'pt1': pt1,
                'nt1': nt1,
                'neut1': neut1,
                'accuracy': F_accuracy
            }
        return render(request, "image.html", context1)


def download_csv(request):

    d = tweets_df
    response = HttpResponse(content_type='text/csv')

    writer = csv.writer(response)

    tweets = tweets_df[['tweet', 'S_Analysis', 'T_Analysis','F_Analysis']]

    writer.writerow(['Tweet', 'Vader Sentiment', 'TextBlob Sentiment','Flair Sentiment'])

    for index, tweet in tweets.iterrows():
        writer.writerow([tweet['tweet'], tweet['S_Analysis'], tweet['T_Analysis'],tweet['F_Analysis']])
   
    response['Content-Disposition'] = 'attachment; filename="tweets.csv"'
    
    return response
