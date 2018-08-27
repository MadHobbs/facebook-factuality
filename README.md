# The Challenge

Build a model to predict the factuality of Facebook news posts from the 2016 Presidential Election and use this model to understand what make a news post likely to be mostly true versus not during the 2016 Presidential Election. 

# What We Did

NOTE: It might be helpful to walk through `results/Predicting Factuality of Facebook Posts.pdf` (our presentation) as you read this because it contains useful vizualisations.

We built off of a dataset from Kaggle which was sourced from Buzzfeed. It contained the URLs of Facebook news posts from the 2016 Presidential Election which were rated by a team of BuzzFeed folks on the following scale: Mostly True, Mixture of True and False, Mostly False, and No Factual Content. We challenged ourselves to build a model to predict the factuality of these Facebook news posts and interpret the feature importances to understand which words were most predictive of each factuality class. After trying some multioutput models and getting poor performance, we decided to opt for a binary prediction problem, predicting "mostly factual" content ("Mostly True" posts) versus "not mostly factual" content ("Mixture of True and False" and "Mosty False" posts). Notice that we excluded "No Factual Content" posts because these posts mainly contained opinion pieces and we wanted to rate posts on a scale of factual to not. It didn't make sense to include posts which weren't even trying to masquerade as fact. 

However, the dataset from Kaggle did not contain the actual text from the news posts. To meet our goal of understanding how certain words were related to the factuality of posts, we were faced with a data joining challenge. We had to scrape Facebook for the content of posts for which we already had the URLs, and merge these with the URLs and metadata we already had from Kaggle.

We scraped the post contents as well as new reaction metadata (i.e.: number of angry reactions, number of sad reactions, etc.) and joined this data to the Kaggle data. 

Before we could model, we had to convert text into usable features. To do this, we first did some pre-processing using NLTK, like removing punctuation and stop words (i.e.: "a", "the", etc.). We then constructed a Bag of Words model where each word became a feature. This produced 5,000 unique words. Using all of these words as predictors would be a terrible idea because we would have more features (5,000 words + metadata) than examples (2,000). For this reason, we the 5,000 words as features through a decision tree and choose the 200 words with the most information gain to keep as our word features in our classification problem. 200 is a relatively arbitrary number, but it ended up working well. We compared this approach with using the 200 most common words and found that selecting features based on information gain yields much higher performing models (understandably so!). 

Because we had imbalanced classes, we decided to weight examples using class weights in the training step. This downweights examples from the more prevalent class and upweights examples from the less prevalent class to mimic a more balanced class scenario. This approach has been shown to make a big difference in terms of model performance, and it certainly did in our case. We also normalized features so that distance metrics when using an SVC, KNN, and Perceptron would work the same across features. 

We cross-validataed a series of models and compared results. Across the board, when analysing feature importance, we saw that posts coming from "mainstream news" (as classified by Buzzfeed) and posts using less emotionally-charged words like "fact," "interview," "fbi," "syrian," "florida," "military," and so forth we predictive of "mostly factual" content. On the other hand, post popularity like number of reactions or shares and emotionally-charged words like "lies," "least," "bad", "hell," and race-based words like "Black" were all indicative of "not mostly factual" content. 

# Dependencies 
- python2.x (I ran on 2.7)
- packages:  NLTK, pandas, tensorflow, sklearn, numpy, urllib2, json, datetime, csv, time

# Directories
- `data+wrangling`
  Code to scrape Facebook API and data acquired thusly along with NLTK wrangling of data and train/test splitting.
  
- `models`
  All of the models we tried along with cross-validation, performance metrics, confusion matrices, and feature importances.
  
- `results`
  The presentation we made to our Machine Learning class at Harvey Mudd College and the project directives from our professor, Dr. Yi-Chieh Wu.

# References
- [Facebook Graph API Reference](https://developers.facebook.com/docs/graph-api/reference/v2.12/post)
- [Kaggle Competition](https://www.kaggle.com/mrisdal/fact-checking-facebook-politics-pages/home), where we sourced the original post urls (without text data) 
