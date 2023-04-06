import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import sys
import csv
#Nelida Schalich-Ayllon, Andrew Rose
#Rotten Tomatoes Critic Reviews
#https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset

stop = stopwords.words('english')

temp = 0

def main():
    # get the command line arguments
    enable_stemming = True
    if len(sys.argv) != 1 and sys.argv[1] != 'YES':
        print("Not ignoring stemming")
    else:
        enable_stemming = False
        print("Ignoring stemming")

    # decide if we want to train the model or not
    val = input("Do you want to retrain the model? (Y/N): ")
    if val == 'Y':
        trainModel(enable_stemming)

    # decide if we want to test the model or not
    val2 = input("Do you want to test the model?(WARNING!! MODEL MAY HAVE BEEN TRAINED WITH STEMMING ENABLED) (Y/N): ")
    if val2 == 'Y':
        testModel(enable_stemming)

    # decide if we want to test the model or not
    userInput(enable_stemming)

def trainModel(enable_stemming):
    # Read in the data
    df = pd.read_csv("rotten_tomatoes_critic_reviews.csv", usecols = ['review_type', 'review_content'])


    #create a naive bayes classifier to classify the reviews
    #create a bag of words model

    positive_words_count = {}
    negative_words_count = {}
    num_positive_reviews = 0
    num_negative_reviews = 0
    total_positive_words = 0
    total_negative_words = 0

    # Split the data into training and testing
    size = len(df.index) * 0.8

    print("Training model is being trained on 80% of the data")
    
    # Building the model
    for index, row in df.iterrows():
        # Manualy set the number of reviews to use for training
        if index >= size:
            break

        # Check if the review is a float - IE Empty
        # TODO Change to reflect jakecs comments
        if type(row['review_content']) == float:
            continue
        
        # Get the class label
        class_label = row['review_type']

        # Iterate through each word in the review
        for word in row['review_content'].split():

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

                # Check if the word is already in the dictionary
                if mword in positive_words_count:
                    positive_words_count[mword] += 1
                else:
                    positive_words_count[mword] = 1
            else:
                num_negative_reviews += 1
                total_negative_words += 1

                # Check if the word is already in the dictionary
                if mword in negative_words_count:
                    negative_words_count[mword] += 1
                else:
                    negative_words_count[mword] = 1



    print("Finished Training model")
    print("Saving model to file")

    # Save the model to a file
    w = csv.writer(open("positive_words_count.csv", "w", encoding="utf-8"))
    for key, val in positive_words_count.items():
        w.writerow([key, val])

    x = csv.writer(open("negative_words_count.csv", "w", encoding="utf-8"))
    for key, val in negative_words_count.items():
        x.writerow([key, val])

    z = csv.writer(open("model_data.csv", "w", encoding="utf-8"))
    z.writerow([num_positive_reviews])
    z.writerow([total_positive_words])
    z.writerow([num_negative_reviews])
    z.writerow([total_negative_words])

    print("Finished saving model to file")
    
def testModel(enable_stemming):
    positive_words_count = {}
    negative_words_count = {}


    # Read in the model
    r = csv.reader(open("positive_words_count.csv", encoding="utf-8"))
    for row in r:
        if row == []:
            continue
        k, v = row
        positive_words_count[k] = int(v)
    

    s = csv.reader(open("negative_words_count.csv", encoding="utf-8"))
    for row in s:
        if row == []:
            continue
        k, v = row
        negative_words_count[k] = int(v)

    t = csv.reader(open("model_data.csv", encoding="utf-8"))
    mylist = list(t)

    # Get the model data
    num_positive_reviews = int(mylist[0][0])
    total_positive_words = int(mylist[2][0])
    num_negative_reviews = int(mylist[4][0])
    total_negative_words = int(mylist[6][0])

    # Read in the data
    df = pd.read_csv("rotten_tomatoes_critic_reviews.csv", usecols = ['review_type', 'review_content'])

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    print("Testing model on 20% of the data")
    test_lower = len(df.index) * 0.8001
    test_lower = int(test_lower)
    test_upper = len(df.index) - 1

    # Testing the model
    for row in range(test_lower, test_upper):
        sentance = df.iloc[row].review_content
        label = df.iloc[row].review_type

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
                true_positive += 1
            else:
                false_positive += 1
        else:
            if label == 'Rotten':
                true_negative += 1
            else:
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


def userInput(enable_stemming):
    # Get user input
    print("Enter a review to classify")
    print("Enter 'q' to quit")

    # Loop until the user quits
    while True:
        review = input()
        if review == 'q':
            break
        classifyReview(review, enable_stemming)

def classifyReview(review, enable_stemming):
    positive_words_count = {}
    negative_words_count = {}
    # Read in the data
    r = csv.reader(open("positive_words_count.csv", encoding="utf-8"))
    # Read in the model
    for row in r:
        if row == []:
            continue
        k, v = row
        positive_words_count[k] = int(v)
    
    s = csv.reader(open("negative_words_count.csv", encoding="utf-8"))
    for row in s:
        if row == []:
            continue
        k, v = row
        negative_words_count[k] = int(v)


    # Get the model data
    t = csv.reader(open("model_data.csv", encoding="utf-8"))
    mylist = list(t)

    num_positive_reviews = int(mylist[0][0])
    total_positive_words = int(mylist[2][0])
    num_negative_reviews = int(mylist[4][0])
    total_negative_words = int(mylist[6][0])

    # initialize the probabilities
    positive = 1
    negitive = 1

    # Iterate through each word in the review
    for word in review.split():

        # Step 1 - Lowercase and remove punctuation
        mword = word.lower()
        mword = mword.strip(string.punctuation)

        # Step 2 - Remove stopwords

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

    print("Positive: ", positive)
    print("Negative: ", negitive)


    # Step 6 - Compare the probabilities and classify the review
    if positive > negitive:
        print("Classified as Fresh/Positive\n")
        
    else:
        print("Classified as Rotten/Negative\n")


if __name__ == "__main__":
    main()