import re
from nltk.corpus import stopwords
import pandas as pd
import time
import numpy as np
import os
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn import metrics

bag_of_words = set()
stop_words = set(stopwords.words('english'))


# Remove non ASCII words
def remove_non_ascii(words):
    return "".join(i for i in words if ord(i) < 128)


# Lemmatize verbs
def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    return ' '.join(lemmatizer.lemmatize(word, pos='v') for word in words.split())


# Remove stopwords
def remove_stopwords(words):
    return ' '.join(word for word in words.split() if word not in stop_words)


# Function to clean and parse reviews and create bag of words
def clean(sent, flag):
    clean_review = remove_non_ascii(sent)
    clean_review = clean_review.lower()
    clean_review = re.sub('[^a-zA-Z]', ' ', clean_review)
    clean_review = re.sub(r'\b\w{1,2}\b', ' ', clean_review)
    clean_review = remove_stopwords(clean_review)
    clean_review = lemmatize_verbs(clean_review)
    if flag == 1 :
        for word in clean_review.split():
            bag_of_words.add(str(word))
    return clean_review


# Function to create feature vector for training and test sentences
def feature_extraction():
    reviews_data_path = 'reviews.csv'

    # Split data into test and train data
    train_df = pd.read_csv(reviews_data_path, sep='|')
    test_df = train_df.iloc[4::5, :]
    train_df = train_df.drop(test_df.index)
    train_df.index = range(len(train_df))
    test_df.index = range(len(test_df))

    print "Split into training and test data"

    for index, row in train_df.iterrows():
        row['text'] = clean(row['text'], 1)

    print "Cleaned training data and created bag of words"

    for index, row in test_df.iterrows():
        row['text'] = clean(row['text'], 0)

    print "Cleaned testing data"

    cv = CountVectorizer(vocabulary = bag_of_words, binary = True)

    embedding_train = cv.fit_transform(train_df['text'])

    print "Embedding for train created"

    embedding_test = cv.transform(test_df['text'])

    print "Embedding for test created"

    train_df['label'] = pd.Categorical(train_df['label'])
    train_df['label'] = train_df['label'].cat.codes

    test_df['label'] = pd.Categorical(test_df['label'])
    test_df['label'] = test_df['label'].cat.codes

    print "Labels encoded"

    return embedding_train, embedding_test, np.array(train_df['label']), np.array(test_df['label'])


# START OF CODE
start_time = time.time()
print("--- %s seconds ---" % (start_time))
embedding_train, embedding_test, label_train, label_test = feature_extraction()

# Create RF model
regr_1 = RandomForestClassifier(n_estimators = 500, random_state = 42)
regr1 = regr_1.fit(embedding_train, label_train)

print "Trained"

# Create pickle of DT model
if os.path.isfile('DTpickle.pkl'):
    os.remove('DTpickle.pkl')
fileObject = open('DTpickle.pkl', 'wb')
pickle.dump(regr1, fileObject)
fileObject.close()

print "Created pickle of model"

# Predict
prediction = regr_1.predict(embedding_test)

print('Accuracy : ', metrics.accuracy_score(label_test,prediction) * 100)

print("--- %s seconds ---" % (time.time() - start_time))
