import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import sys
import csv
#Nelida Schalich-Ayllon, Andrew Rose
#Rotten Tomatoes Critic Reviews
#https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset
#PLACE DATASET IN /SRC

stop = stopwords.words('english')

def main():
    # get the command line arguments
    enable_stemming = True
    if len(sys.argv) == 2 and sys.argv[1] == 'YES':
        enable_stemming = False
        print("Ignoring stemming")
    else:
        print("Not ignoring stemming")

    # decide if we want to train the model or not
    train_val = input("Do you want to train the model? [y/n]: ")
    train_val = train_val.lower()
    if train_val == 'y' or train_val == 'yes':
        train(enable_stemming)

    # decide if we want to test the model or not
    print()
    test_val = input("Do you want to test the model? [y/n]: ")
    test_val = test_val.lower()
    if test_val == 'y' or test_val == 'yes':
        test(enable_stemming)

    # ask user if they want to enter a review to be classified
    userInput(enable_stemming)


def train(stem):
    # Read in the data
    df = pd.read_csv("rotten_tomatoes_critic_reviews.csv", usecols = ['review_type', 'review_content'])
    # Split the data into training and testing
    size = len(df.index) * 0.8
    print("Training on 80% of the data")

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
        # Checking if at upper bound so that training is on only 80% of the data
        if index >= size:
            break

        # Get the class label
        label = row['review_type']

        # Get the review content
        review = row['review_content']

        # Check if the review is a float - IE Empty
        if type(review) == float:
            continue

        # Iterate through each word in the review
        for w in review.split():

            # Step 1 - Lowercase and remove punctuation
            word = w.lower()
            word = word.strip(string.punctuation)

            # Step 2 - Remove stopwords
            if word in stop:
                continue
            
            # Step 3 - Stemming
            if stem:
                word = PorterStemmer().stem(word)

            # Step 4 - Add to the dictionary
            if label == 'Fresh':
                num_positive_reviews += 1
                total_positive_words += 1

                # Check if the word is already in the dictionary
                if word in positive_words_count:
                    positive_words_count[word] += 1
                else:
                    positive_words_count[word] = 1
            else:
                num_negative_reviews += 1
                total_negative_words += 1

                # Check if the word is already in the dictionary
                if word in negative_words_count:
                    negative_words_count[word] += 1
                else:
                    negative_words_count[word] = 1

    print("Training complete")
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

    print("Model saved to file")
    

def test(stem):
    positive_words_count = {}
    negative_words_count = {}

    # Read in the model
    r = csv.reader(open("positive_words_count.csv", encoding="utf-8"))
    for row in r:
        if row == []:
            continue
        positive_words_count[row[0]] = int(row[1])
    

    s = csv.reader(open("negative_words_count.csv", encoding="utf-8"))
    for row in s:
        if row == []:
            continue
        negative_words_count[row[0]] = int(row[1])

    t = csv.reader(open("model_data.csv", encoding="utf-8"))
    mylist = list(t)

    # Get the model data
    num_positive_reviews = int(mylist[0][0])
    total_positive_words = int(mylist[1][0])
    num_negative_reviews = int(mylist[2][0])
    total_negative_words = int(mylist[3][0])

    # Read in the data
    df = pd.read_csv("rotten_tomatoes_critic_reviews.csv", usecols = ['review_type', 'review_content'])

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    print("Testing model on 20% of the data")
    print()
    start = int(len(df.index) * 0.8001)

    # Testing the model
    for row in range(start, len(df.index) - 1):
        label = df.iloc[row].review_type
        review = df.iloc[row].review_content

        # Check if the review is a float - IE Empty
        if type(review) == float:
            continue

        # initialize the probabilities
        positive_prob = 1
        negative_prob = 1

        # Iterate through each word in the review
        for w in review.split():

            # Step 1 - Lowercase and remove punctuation
            word = w.lower()
            word = word.strip(string.punctuation)

            # Step 2 - Remove stopwords
            #TODO: Enable this
            if word in stop:
                continue


            # Step 3 - Stemming
            if stem:
                word = PorterStemmer().stem(word)

            # Step 4 - Calculate the probability of the word for each class, with smoothing
            positive_prob *= (positive_words_count.get(word, 1) + 1) / (total_positive_words + len(positive_words_count))
            negative_prob *= (negative_words_count.get(word, 1) + 1) / (total_negative_words + len(negative_words_count))

        # Step 5 - Calculate the probability of the class
        positive_prob *= num_positive_reviews / (num_positive_reviews + num_negative_reviews)
        negative_prob *= num_negative_reviews / (num_positive_reviews + num_negative_reviews)
        # Step 6 - Compare the probabilities and classify the review
        if positive_prob > negative_prob:
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
    print("Number of true positives:", true_positive)
    print("Number of true negatives:", true_negative)
    print("Number of false positives:", false_positive)
    print("Number of false negatives:", false_negative)

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f_score = 2 * precision * recall / (precision + recall)
    specificity = true_negative / (true_negative + false_positive)
    npv = true_negative / (true_negative + false_negative)

    print("Sensitivity (recall):", recall)
    print("Specificity:", specificity)
    print("Precision:", precision)
    print("Negative predictive value:", npv)
    print("Accuracy:", accuracy)
    print("F-Score:", f_score)
    print()


def userInput(stem):
    # Loop until the user quits
    while True:
        # Get user input
        print("\nEnter your sentence: ")
        print("Enter 'q' to quit")
        review = input()
        if review == 'q':
            break
        print()
        print("Sentence S: ")
        print("'", review, "'")
        classifyReview(review, stem)


def classifyReview(review, stem):
    positive_words_count = {}
    negative_words_count = {}
    # Read in the data
    r = csv.reader(open("positive_words_count.csv", encoding="utf-8"))
    # Read in the model
    for row in r:
        if row == []:
            continue
        positive_words_count[row[0]] = int(row[1])
    
    s = csv.reader(open("negative_words_count.csv", encoding="utf-8"))
    for row in s:
        if row == []:
            continue
        negative_words_count[row[0]] = int(row[1])


    # Get the model data
    t = csv.reader(open("model_data.csv", encoding="utf-8"))
    mylist = list(t)

    num_positive_reviews = int(mylist[0][0])
    total_positive_words = int(mylist[1][0])
    num_negative_reviews = int(mylist[2][0])
    total_negative_words = int(mylist[3][0])

    # initialize the probabilities
    positive_prob = 1
    negative_prob = 1

    # Iterate through each word in the review
    for w in review.split():

        # Step 1 - Lowercase and remove punctuation
        word = w.lower()
        word = word.strip(string.punctuation)

        # Step 2 - Remove stopwords
        if word in stop:
                continue

        # Step 3 - Stemming
        if stem:
            word = PorterStemmer().stem(word)
        
        # Step 4 - Calculate the probability of the word for each class, with smoothing
        positive_prob *= (positive_words_count.get(word, 1) + 1) / (total_positive_words + len(positive_words_count))
        negative_prob *= (negative_words_count.get(word, 1) + 1) / (total_negative_words + len(negative_words_count))

    # Step 5 - Calculate the probability of the class
    positive_prob *= num_positive_reviews / (num_positive_reviews + num_negative_reviews)
    negative_prob *= num_negative_reviews / (num_positive_reviews + num_negative_reviews)

    # Step 6 - Compare the probabilities and classify the review
    if positive_prob > negative_prob:
        print("was classified as 'Fresh'\n")
        
    else:
        print("was classified as 'Rotten'\n")

    print("P('Fresh' | S) = ", positive_prob)
    print("P('Rotten' | S) = ", negative_prob)


if __name__ == "__main__":
    main()