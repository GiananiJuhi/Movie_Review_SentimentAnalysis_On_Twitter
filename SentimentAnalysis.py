# -*- coding: utf-8 -*-

import nltk
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt

nltk.download('stopwords')

nltk.download("movie_reviews")

from nltk import ngrams
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords 
import string
import re

stopwords_english = stopwords.words('english')

# clean words, i.e. remove stopwords and punctuation
def clean_words(words, stopwords_english):
    words_clean = []
    for word in words:
        word = word.lower()
        if word not in stopwords_english and word not in string.punctuation:
            words_clean.append(word)
      
    return words_clean

from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()
from bs4 import BeautifulSoup

pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))

#This function is used to remove hyperlinks , all text preceding with @ symbol etc
def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessary white spaces,
    # I will tokenize and join together to remove unnecessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()

# feature extractor function for unigram
def bag_of_words(words):    
    words_dictionary = dict([word, True] for word in words)    
    return words_dictionary

# feature extraction function for bigram
def bag_of_ngrams(words, n=2):
    words_ng = []
    for item in iter(ngrams(words, n)):
        words_ng.append(item)
    words_dictionary = dict([word, True] for word in words_ng)    
    return words_dictionary

from nltk.tokenize import word_tokenize
nltk.download('punkt')

#Examples
print("Let us understand the concept with the following example")
text = "It was a very good movie."
print(text)
words = word_tokenize(text.lower())
print("Breaking this text in the form of tokens")
print (words)

# working with cleaning words
# i.e. removing stopwords and punctuation
words_clean = clean_words(words, stopwords_english)
print("Removing stopwords and punctuation")
print (words_clean)

important_words = ['above', 'below', 'off', 'over', 'under', 'more', 'most', 'such', 'no', 'nor', 'not', 'only', 'so', 'than', 'too', 'very', 'just', 'but']
 
stopwords_english_for_bigrams = set(stopwords_english) - set(important_words)
 
words_clean_for_bigrams = clean_words(words, stopwords_english_for_bigrams)
print (words_clean_for_bigrams)

# We will use general stopwords for unigrams 
# And special stopwords list for bigrams
unigram_features = bag_of_words(words_clean)
print (unigram_features)

bigram_features = bag_of_ngrams(words_clean_for_bigrams)
print (bigram_features)

all_features = unigram_features.copy()
all_features.update(bigram_features)
print (all_features)

def bag_of_all_words(words, n=2):
    words_clean = clean_words(words, stopwords_english)
    words_clean_for_bigrams = clean_words(words, stopwords_english_for_bigrams)
 
    unigram_features = bag_of_words(words_clean)
    bigram_features = bag_of_ngrams(words_clean_for_bigrams)
 
    all_features = unigram_features.copy()
    all_features.update(bigram_features)
 
    return all_features
 
print (bag_of_all_words(words))

from nltk.corpus import movie_reviews

pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append(words)
 
neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append(words)

# positive reviews feature set
pos_reviews_set = []
for words in pos_reviews:
    pos_reviews_set.append((bag_of_all_words(words), 'pos'))
 
# negative reviews feature set
neg_reviews_set = []
for words in neg_reviews:
    neg_reviews_set.append((bag_of_all_words(words), 'neg'))

print("Total number of Positive And Negative reviews in Our Dataset")
print (len(pos_reviews_set), len(neg_reviews_set)) # Output: (1000, 1000)
 
# radomize pos_reviews_set and neg_reviews_set
# doing so will output different accuracy result everytime we run the program
from random import shuffle 
shuffle(pos_reviews_set)
shuffle(neg_reviews_set)
 
test_set = pos_reviews_set[:200] + neg_reviews_set[:200]
train_set = pos_reviews_set[200:] + neg_reviews_set[200:]

print("Dividing our Training And Testing Dataset as follows")
print(len(test_set),  len(train_set)) # Output: (400, 1600)

from nltk import classify
from nltk import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(train_set)
 
accuracy = classify.accuracy(classifier, test_set)
print("According to the division the Accuracy ")
print(accuracy)
 
print (classifier.show_most_informative_features(10))

custom_review = "I hated the film. It was a disaster. Poor direction, bad acting."
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = bag_of_all_words(custom_review_tokens)
print (classifier.classify(custom_review_set))

# probability result
prob_result = classifier.prob_classify(custom_review_set)
print (prob_result) # Output: <ProbDist with 2 samples>
print (prob_result.max()) # Output: neg
print (prob_result.prob("neg"))
print (prob_result.prob("pos"))

import sys, tweepy

ConsumerKey ="#Your_ConsumerKey"
ConsumerSecret ="#Your_ConsumerSecret"
accessToken ="#Your_accessToken"
accessTokenSecret ="#Your_accessTokenSecret"
auth = tweepy.OAuthHandler(ConsumerKey, ConsumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)

text = input("Enter movie name: ")
public_tweets = api.search(text)

x=0
test_result = []
for tweet in public_tweets:
    test_result.append(tweet_cleaner(tweet.text))
    x=x+1

print("Printing the Tweets")
print(test_result)
print("Total number of tweets are ")
print (x)

negative_review=0
positive_review=0
for t in test_result:
  custom_review_tokens = word_tokenize(t)
  custom_review_set = bag_of_all_words(custom_review_tokens)
  prob_result = classifier.prob_classify(custom_review_set)
  negative_review=(prob_result.prob("neg")+prob_result.prob("neg"))
  positive_review=(prob_result.prob("pos")+prob_result.prob("pos"))

print (negative_review)
print (positive_review)
print ('Review : '+prob_result.max())

import matplotlib.pyplot as plt

negative=(negative_review/2)*100
positive=(positive_review/2)*100

negative = format(negative, ".2f")
positive = format(positive, '.2f')

labels=['Positive['+str(positive)+'%]','Negative['+str(negative)+'%]']
sizes = [positive,negative]
colors = ['yellow','red']
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
plt.legend(patches,labels,loc='best')
plt.title('How many people are reacting on '+text+' Tweets.')
plt.axis('equal')
plt.tight_layout()
plt.show()