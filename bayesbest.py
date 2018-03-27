import math, os, pickle, re
import string

class Bayes_Classifier:
    '''Implements an improved Naive Bayes classifer designed to classify movie
    reviews as either positive or negative, based on the words within the review.
    This class is adapted upon bayes.py, byt modifying the neutrality bias, and
    adding extra checks for punctuation and review length.'''

    def __init__(self, trainDirectory = "movie_reviews/"):
        '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
        cache of a trained classifier has been stored, it loads this cache.  Otherwise,
        the system will proceed through training.  After running this method, the classifier
        is ready to classify input text.'''

        self._trainDirectory = trainDirectory

        # "Bayes Best" Tweaking Parameters:
        self._neutralityBias     = 0.01  # "Neutral" Buffer between Positive and Negative Reviews
        self._punctuationWeight  = 1000  # Weight to apply to punctuation, in Training Data
        self._shortReviewWeight  = 0.1   # Weight to apply to Short Reviews, in Training Data
        self._shortReviewLength  = 15    # Cut-off for what is considered a "Short Review"
        self._reviewLengthWeight = 0.12  # Bias to apply to Conditional Prob based on Sentence Length

        # Load pickled Training Object (if it exists)
        if (os.path.exists("database")):
            database = self.load("database")
            self._positiveWords    = database[0] # Dictionary of word counts in POSITIVE reviews
            self._negativeWords    = database[1] # Dictionary of word counts in NEGATIVE reviews
            self._numPositiveDocs  = database[2] # Number of POSITIVE reviews
            self._numNegativeDocs  = database[3] # Number of NEGATIVE reviews
            self._numPositiveWords = database[4] # Number of total words in all POSITIVE reviews
            self._numNegativeWords = database[5] # Number of total words in all NEGATIVE reviews
        else:
            self.train()


    @property
    def trainDirectory(self):
        '''Getter for the trainDirectory property'''
        return self._trainDirectory

    @property
    def positiveWords(self):
        '''Getter for the positiveWords property'''
        return self._positiveWords

    @property
    def negativeWords(self):
        '''Getter for the negativeWords property'''
        return self._negativeWords

    @property
    def numPositiveDocs(self):
        '''Getter for the numPositiveDocs property'''
        return self._numPositiveDocs

    @property
    def numNegativeDocs(self):
        '''Getter for the numNegativeDocs property'''
        return self._numNegativeDocs

    @property
    def numPositiveWords(self):
        '''Getter for the numPositiveWords property'''
        return self._numPositiveWords

    @property
    def numNegativeWords(self):
        '''Getter for the numNegativeWords property'''
        return self._numNegativeWords

    @property
    def neutralityBias(self):
        '''Getter for the neutralityBias property'''
        return self._neutralityBias

    @property
    def punctuationWeight(self):
        '''Getter for the punctuationWeight property'''
        return self._punctuationWeight

    @property
    def shortReviewWeight(self):
        '''Getter for the shortReviewWeight property'''
        return self._shortReviewWeight

    @property
    def shortReviewLength(self):
        '''Getter for the shortReviewLength property'''
        return self._shortReviewLength

    @property
    def reviewLengthWeight(self):
        '''Getter for the reviewLengthWeight property'''
        return self._reviewLengthWeight

    @trainDirectory.setter
    def trainDirectory(self, value):
        '''Setter for the trainDirectory property'''
        self._trainDirectory = value

    @positiveWords.setter
    def positiveWords(self, value):
        '''Setter for the positiveWords property'''
        self._positiveWords = value

    @negativeWords.setter
    def negativeWords(self, value):
        '''Setter for the negativeWords property'''
        self._negativeWords = value

    @numPositiveDocs.setter
    def numPositiveDocs(self, value):
        '''Setter for the numPositiveDocs property'''
        self._numPositiveDocs = value

    @numNegativeDocs.setter
    def numNegativeDocs(self, value):
        '''Setter for the numNegativeDocs property'''
        self._numNegativeDocs = value

    @numPositiveWords.setter
    def numPositiveWords(self, value):
        '''Setter for the numPositiveWords property'''
        self._numPositiveWords = value

    @numNegativeWords.setter
    def numNegativeWords(self, value):
        '''Setter for the numNegativeWords property'''
        self._numNegativeWords = value

    @neutralityBias.setter
    def neutralityBias(self, value):
        '''Setter for the neutralityBias property'''
        self._neutralityBias = value

    @punctuationWeight.setter
    def punctuationWeight(self, value):
        '''Setter for the punctuationWeight property'''
        self._punctuationWeight = value

    @shortReviewWeight.setter
    def shortReviewWeight(self, value):
        '''Setter for the shortReviewWeight property'''
        self._shortReviewWeight = value

    @shortReviewLength.setter
    def shortReviewLength(self, value):
        '''Setter for the shortReviewLength property'''
        self._shortReviewLength = value

    @reviewLengthWeight.setter
    def reviewLengthWeight(self, value):
        '''Setter for the reviewLengthWeight property'''
        self._reviewLengthWeight = value


    def train(self):
        '''Trains the Naive Bayes Sentiment Classifier.'''

        # Get List of Files to Train On
        lFileList = []
        for fFileObj in os.walk(self.trainDirectory):
            lFileList = fFileObj[2]
            break

        # Initialize the Positive & Negative Word DICTIONARIES
        self.positiveWords = {}
        self.negativeWords = {}

        # Initialize the Positive & Negative Word COUNTERS
        self.numPositiveDocs  = 0
        self.numNegativeDocs  = 0
        self.numPositiveWords = 0
        self.numNegativeWords = 0

        # Train (i.e. Fill the Dictionaries and Word Counters)
        for fname in lFileList:

            with open(self.trainDirectory + fname) as fh:
                words = self.tokenize(fh.read())

            # Process Positive Review:
            if (fname.split('-')[1] == "5"):

                # Update Number of Docs (Reviews)
                self.numPositiveDocs = self.numPositiveDocs + 1

                # Update (Scaled) Unigram and Total Word Counters
                for word in words:
                    word = word.lower()
                    if word in string.punctuation:
                        self.numPositiveWords = self.numPositiveWords + self.punctuationWeight
                        self.positiveWords[word] = self.positiveWords.get(word,0) + self.punctuationWeight
                    elif len(words) < self.shortReviewLength:
                        self.numPositiveWords = self.numPositiveWords + self.shortReviewWeight
                        self.positiveWords[word] = self.positiveWords.get(word,0) + self.shortReviewWeight
                    else:
                        self.numPositiveWords = self.numPositiveWords + 1
                        self.positiveWords[word] = self.positiveWords.get(word,0) + 1

            # Process Negative Review:
            if (fname.split('-')[1] == "1"):

                # Update Number of Docs (Reviews)
                self.numNegativeDocs = self.numNegativeDocs + 1

                # Update (Scaled) Unigram and Total Word Counters
                for word in words:
                    word = word.lower()
                    if word in string.punctuation:
                        self.numNegativeWords = self.numNegativeWords + self.punctuationWeight
                        self.negativeWords[word] = self.negativeWords.get(word,0) + self.punctuationWeight
                    elif len(words) < self.shortReviewLength:
                        self.numNegativeWords = self.numNegativeWords + self.shortReviewWeight
                        self.negativeWords[word] = self.negativeWords.get(word,0) + self.shortReviewWeight
                    else:
                        self.numNegativeWords = self.numNegativeWords + 1
                        self.negativeWords[word] = self.negativeWords.get(word,0) + 1

        # Save Results of the Training
        self.save([self.positiveWords, self.negativeWords,
                   self.numPositiveDocs, self.numNegativeDocs,
                   self.numPositiveWords, self.numNegativeWords], "database")


    def classify(self, sText):
        '''Given a target string sText, this function returns the most likely document
        class to which the target string belongs. This function should return one of three
        strings: "positive", "negative" or "neutral".'''

        # Calculate Sum of Logs of Conditional Probabilities
        sumlog_p_fi_positive = 0.0
        sumlog_p_fi_negative = 0.0

        for word in self.tokenize(sText):

            word = word.lower()

            cond_prob_positive = math.log10(float(self.positiveWords.get(word,0) + 1) / float(self.numPositiveWords + 1))
            sumlog_p_fi_positive = sumlog_p_fi_positive + cond_prob_positive

            cond_prob_negative = math.log10(float(self.negativeWords.get(word,0) + 1) / float(self.numNegativeWords + 1))
            sumlog_p_fi_negative = sumlog_p_fi_negative + cond_prob_negative

        # Calculate Log of Prior Probabilities of Positive and Negative Classes
        log_prior_prob_positive = math.log10(float(self.numPositiveDocs) / float(self.numPositiveDocs + self.numNegativeDocs))
        log_prior_prob_negative = math.log10(float(self.numNegativeDocs) / float(self.numPositiveDocs + self.numNegativeDocs))

        # Calculate the Log of Final Probabilities
        log_final_prob_positive = sumlog_p_fi_positive + log_prior_prob_positive
        log_final_prob_negative = sumlog_p_fi_negative + log_prior_prob_negative

        # Check for Delta in Final Probabilities (minus Scaled Threshold)
        delta_prob = (log_final_prob_positive - log_final_prob_negative) + (len(self.tokenize(sText)) * self.reviewLengthWeight)

        if (delta_prob > +1.0 * self.neutralityBias):
            return "positive"
        if (delta_prob < -1.0 * self.neutralityBias):
            return "negative"
        else:
            return "neutral"


    def loadFile(self, sFilename):
        '''Given a file name, return the contents of the file as a string.'''

        f = open(sFilename, "r")
        sTxt = f.read()
        f.close()
        return sTxt

    def save(self, dObj, sFilename):
        '''Given an object and a file name, write the object to the file using pickle.'''

        f = open(sFilename, "w")
        p = pickle.Pickler(f)
        p.dump(dObj)
        f.close()

    def load(self, sFilename):
        '''Given a file name, load and return the object stored in the file.'''

        f = open(sFilename, "r")
        u = pickle.Unpickler(f)
        dObj = u.load()
        f.close()
        return dObj

    def tokenize(self, sText):
        '''Given a string of text sText, returns a list of the individual tokens that
        occur in that string (in order).'''

        lTokens = []
        sToken = ""
        for c in sText:
            if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
                sToken += c
            else:
                if sToken != "":
                    lTokens.append(sToken)
                    sToken = ""
                if c.strip() != "":
                    lTokens.append(str(c.strip()))

        if sToken != "":
            lTokens.append(sToken)

        return lTokens
