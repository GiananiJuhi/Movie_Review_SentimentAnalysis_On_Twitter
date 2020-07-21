# Movie Review Sentiment Analysis 

This project classifies the sentiments of tweets on movie reviews as positive or negative and shows the visual representation of the same

## Prerequisites
<ol>
<li> Install "tweepy" module with command:

        pip install tweepy

<li> Install "numpy" module with command:

        pip3 install numpy
		
	Note above command works for Python 3.x versions

<li> Install "matplotlib" module with command:

        pip install matplotlib

<li> Install "nltk" module with command:

        pip3 install nltk
		
	Note above command works for Python 3.x versions

<li> Install "pandas" module with command:

        pip install pandas

</ol>

## How to run the project
<ol>
<li> Clone the project and open SentimentAnalysis.py file.
<li> You need to create a Twitter Developer account. After creating Twitter Developer account, you will get 4 keys: API key, API Secret key, Access token, Access token secret.
<li> Copy these 4 keys on line numbers 179, 180, 181 and 182 respectively in SentimentAnalysis.py file.
<li> On line 182, Enter the movie name
<li> Save the changes and run SentimentAnalysis.py file
<li> The visual representation will be shown as an output on the terminal
</ol>

## Note
The program extracts real time streaming tweets from Twitter. Sometime it may happen that it is not printing any desired output.
That is because it is not getting the real time tweets from twitter , in such cases re-run the code .
Also it might happen that the output varies each time you run the code that is because it is fetching real time tweets from twitter .