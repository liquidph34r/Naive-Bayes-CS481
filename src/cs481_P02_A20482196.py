import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import sys
#Nelida Schalich-Ayllon, Andrew Rose
#Rotten Tomatoes Critic Reviews
#https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset


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
    
    for index, row in df.iterrows():
        #print(row['review_type'], row['review_content'])
        if type(row['review_content']) == float:
            #print("CONTINUE")
            continue
        
        class_label = row['review_type']

        #print(type (row['review_content']))
        for word in row['review_content'].split():
            #print(word)

            # Step 1 - Lowercase and remove punctuation
            mword = word.lower()
            mword = mword.strip(string.punctuation)

            # Step 2 - Remove stopwords
            if mword in stopwords.words('english'):
                continue
            
            # Step 3 - Stemming
            if enable_stemming:
                stemmer = PorterStemmer()
                mword = stemmer.stem(mword)

            if class_label == 'fresh':
                if mword in positive_words_count:
                    positive_words_count[mword] += 1
                else:
                    positive_words_count[mword] = 1
            else:
                if mword in negative_words_count:
                    negative_words_count[mword] += 1
                else:
                    negative_words_count[mword] = 1
            print(mword)
        
    print(positive_words_count)
    print(negative_words_count)


if __name__ == "__main__":
    main()