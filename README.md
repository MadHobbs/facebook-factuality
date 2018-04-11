# ML_Project

# requirement: 
- python2.x (I ran on 2.7)
- packages:  urllib2 ,json,datetime,csv, time,pandas

# Files
- facebook-fact-check.csv
  original dataset
- facebook_statuses.csv
  Scraped information about status.
- posts.py
  This code is to scrape information about the post and creat a csv file
- validations.py
  Code for validating the classifier. ()
- bag_of_words.py
  Code for creating bag of words. 
- util.py
  - Reading csv files
  - everything related to data aquisitino (format etc)

# References
- Facebook Graph API (Post)
  https://developers.facebook.com/docs/graph-api/reference/v2.12/post

# TODO
- Try high variance/high bias models.
    - 
- Know about the dataset balance well. 
- User Stratified 
- Frorcus on one message
- One redommendation -> NLTK can do the natural language processing
- Use more common feature extractoin -> surpsing words (words that don't show up in other sets)
- Don't use all of the bags. You could calculate the entropy and use the highest 200 entrpies
- Stay away from other kinds of prediction
- Show confusion matrix 
- Use Normalization 
- Take a look at the rubic
- Amount of the insight you get from the model is important
    - look at the coefficents/ error analysis