# Movie_Review_Classifier
This project examines the creation and tuning of a Naive Bayes Classifier that trains on movie review (and other review) logs collected from _www.rateitall.com_.  The movies are now a few years old, but the same approach could be used on any similar movie data set -- regardless of the the movie vintage.  This was written as a HW assignment for CS510 (Artificial Intelligence) @ Drexel University, Fall 2018.

### Summary:

There are two parts to the project:

- The first phase creates a simple Naive-Bayes classifier that trains on the frequency of each single word in the positive reviews (reviews receiving 5 out of 5 stars), and the negative review (reviews receiving only 1 out of 5 stars).  After training, an input sentence is classified using "unigram" classification, which compares the frequency of each word in the input sentence against the frequencies of the same word in the positive and negative reviews.  If the words appear more often in the positive movie reviews, then the input sentence is classified as "Positive", and if the words appear more often in the negative movie review, then the review is classified as "Negative".  If it's too close to call, the input sentence is deemed to be "Neutral".  Code for this phase is included in __bayes.py__.

- The second phase attempts to improve upon on this classifier, taking into account various weaknesses observed in the initial classification.  Code for the second phase is included in __bayesbest.py__.

NOTE: This project was originally created with much larger training and testing data sets, but for simplicity / transfer speed, only a subset are posted here to Github.  For more information, please refer to the original source data from _rateitall.com_.

### Instructions:

The code runs in Python 2.  Execute the evaluate.py file first.  To test out the second phase, change the input file name in evaluate.py, and then re-run it.  Cheers!
