# The Challenge

Build a model to predict the factuality of Facebook news posts from the 2016 Presidential Election and use this model to understand what make a news post likely to be mostly true versus not during the 2016 Presidential Election. 

# What We Did

We built off of a dataset from Kaggle which was sourced from Buzzfeed. It contained the URLs of Facebook news posts from the 2016 Presidential Election which were rated by a team of BuzzFeed folks on the following scale: Mostly True, Mixture of True and False, Mostly False, and No Factual Content. We challenged ourselves to build a multiclass model to predict the factuality of these Facebook news posts and interpret the feature importances to understand which words were most predictive of each factuality class. 

However, the dataset from Kaggle did not contain the actual text from the news posts. To meet our goal of understanding how certain words were related to the factuality of posts, we were faced with a data joining challenge. We had to scrape Facebook for the content of posts for which we already had the URLs, and merge these with the URLs and metadata we already had from Kaggle.

We scraped the post contents as well as new reaction metadata (i.e.: number of angry reactions, number of sad reactions, etc.) and joined this data to the Kaggle data. 

Before we could model, we had to convert text into usable features. To do this, we did some pre-processing using NLTK and then constructed a Bag of Words model.

(to be continued)

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
