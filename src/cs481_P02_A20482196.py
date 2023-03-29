import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import sys
#Nelida Schalich-Ayllon, Andrew Rose
#Rotten Tomatoes Critic Reviews
#https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset

stop = stopwords.words('english')

temp = 0

def main():
    #print("Hello World")
    #read in the csv file
    df = pd.read_csv("rotten_tomatoes_critic_reviews.csv", usecols = ['review_type', 'review_content'])

    enable_stemming = True
    if len(sys.argv) != 1 and sys.argv[1] != 'YES':
        #print(len(sys.argv))
        print("Not ignoring stemming")
        
    else:
        enable_stemming = False
        print("Ignoring stemming")

    #print(df.head())

    #create a naive bayes classifier to classify the reviews
    #create a bag of words model

    positive_words_count = {}
    negative_words_count = {}
    num_positive_reviews = 0
    num_negative_reviews = 0
    total_positive_words = 0
    total_negative_words = 0
    
    # Building the model
    for index, row in df.iterrows():
        # Manualy set the number of reviews to use for training
        if index == 500000:
            break

        #print(row['review_type'], row['review_content'])

        # Check if the review is a float - IE Empty
        if type(row['review_content']) == float:
            continue
        
        # Get the class label
        class_label = row['review_type']

        #print(index)

        #print(type (row['review_content']))

        # Iterate through each word in the review
        for word in row['review_content'].split():
            #print(word)

            # Step 1 - Lowercase and remove punctuation
            mword = word.lower()
            mword = mword.strip(string.punctuation)

            # Step 2 - Remove stopwords
            #TODO: Figure out how to speed up?
            #if mword in stop:
            #    continue
            
            # Step 3 - Stemming
            if enable_stemming:
                stemmer = PorterStemmer()
                mword = stemmer.stem(word)

            # Step 4 - Add to the dictionary
            if class_label == 'Fresh':
                num_positive_reviews += 1
                total_positive_words += 1
                #print("increase pos")
                # Check if the word is already in the dictionary
                if mword in positive_words_count:
                    positive_words_count[mword] += 1
                else:
                    positive_words_count[mword] = 1
            else:
                num_negative_reviews += 1
                total_negative_words += 1
                #print("increase neg")
                if mword in negative_words_count:
                    negative_words_count[mword] += 1
                else:
                    negative_words_count[mword] = 1


    print("Finished Training model")

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    # Testing the model
    for row in range(500001, 600000):
        # Manualy set the number of reviews to use for testing
        sentance = df.iloc[row].review_content
        label = df.iloc[row].review_type

        #print(total_positive_words)

        # initialize the probabilities
        positive = 1
        negitive = 1

        # Check if the review is a float - IE Empty
        if type(sentance) == float:
            continue

        # Iterate through each word in the review
        for word in sentance.split():

            # Step 1 - Lowercase and remove punctuation
            mword = word.lower()
            mword = mword.strip(string.punctuation)

            # Step 2 - Remove stopwords
            #TODO: Enable this


            # Step 3 - Stemming
            if enable_stemming:
                stemmer = PorterStemmer()
                mword = stemmer.stem(word)

            # Step 4 - Calculate the probability of the word for each class
            positive *= (positive_words_count.get(mword, 1)/total_positive_words)
            negitive *= (negative_words_count.get(mword, 1)/total_negative_words)

        # Step 5 - Calculate the probability of the class
        positive *= (num_positive_reviews/(num_positive_reviews + num_negative_reviews))
        negitive *= (num_negative_reviews/(num_positive_reviews + num_negative_reviews))

        # Step 6 - Compare the probabilities and classify the review
        if positive > negitive:
            if label == 'Fresh':
                #print("Correct")
                true_positive += 1
            else:
                #print("Incorrect")
                false_positive += 1
        else:
            if label == 'Rotten':
                #print("Correct")
                true_negative += 1
            else:
                #print("Incorrect")
                false_negative += 1

    # Step 7 - Calculate the accuracy and other metrics
    print("True Positive: ", true_positive)
    print("False Positive: ", false_positive)
    print("True Negative: ", true_negative)
    print("False Negative: ", false_negative)

    print("Accuracy: ", (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative))
    print("misclassification rate: ", (false_positive + false_negative)/(true_positive + true_negative + false_positive + false_negative))
    print("Recall(sensitivity): ", true_positive/(true_positive + false_negative))
    print("Specificity: ", true_negative/(true_negative + false_positive))
    print("Precision: ", true_positive/(true_positive + false_positive))
    print("negative predictive value: ", true_negative/(true_negative + false_negative))
    print("f-score: ", (2*true_positive)/((2*true_positive) + false_positive + false_negative))
   # hello git testing


if __name__ == "__main__":
    main()