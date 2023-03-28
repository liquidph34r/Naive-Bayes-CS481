import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import sys
#Nelida Schalich-Ayllon, Andrew Rose
#Rotten Tomatoes Critic Reviews
#https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset

stop = stopwords.words('english')
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
    
    #print(df.iloc[1000000].review_content)
    for index, row in df.iterrows():
        if index == 100000:
            break
        #print(row['review_type'], row['review_content'])
        if type(row['review_content']) == float:
            #print("CONTINUE")
            continue
        
        class_label = row['review_type']
        #print(index)

        #print(type (row['review_content']))
        for word in row['review_content'].split():
            #print(word)

            # Step 1 - Lowercase and remove punctuation
            mword = word.lower()
            mword = mword.strip(string.punctuation)

            # Step 2 - Remove stopwords
            #if mword in stop:
            #    continue
            
            # Step 3 - Stemming
            if enable_stemming:
                stemmer = PorterStemmer()
                mword = stemmer.stem(word)

            if class_label == 'Fresh':
                num_positive_reviews += 1
                total_positive_words += 1
                #print("increase pos")
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
            #print(mword)
        
    #print(positive_words_count)
    #print(negative_words_count)

    #print("starting sum")

    print("finished sum")

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0


    for row in range(100001, 120000):
        sentance = df.iloc[row].review_content
        label = df.iloc[row].review_type

        #print(total_positive_words)
        positive = 1
        negitive = 1

        if type(sentance) == float:
            continue

        for word in sentance.split():
            mword = word.lower()
            mword = mword.strip(string.punctuation)

            if enable_stemming:
                stemmer = PorterStemmer()
                mword = stemmer.stem(word)

            positive *= (positive_words_count.get(mword, 1)/total_positive_words)
            negitive *= (negative_words_count.get(mword, 1)/total_negative_words)

        positive *= (num_positive_reviews/(num_positive_reviews + num_negative_reviews))
        negitive *= (num_negative_reviews/(num_positive_reviews + num_negative_reviews))

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
   


if __name__ == "__main__":
    main()