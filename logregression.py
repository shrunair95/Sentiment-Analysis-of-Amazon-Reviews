import re
from nltk.corpus import stopwords
import pandas as pd
import time
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import os

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
def create_model():
    reviews_data_path = 'reviews.csv'

    # Read data from file
    train_df = pd.read_csv(reviews_data_path, sep='|')

    print "Read data into dataframe"

    # Clean data
    for index, row in train_df.iterrows():
        row['text'] = clean(row['text'], 1)

    print "Cleaned data"

    # Create BOW embedding of text
    cv = CountVectorizer(vocabulary=bag_of_words, binary = True)

    embedding_train = cv.fit_transform(train_df['text'])

    print "Embedding created"

    # Encode labels to 0 and 1
    train_df['label'] = pd.Categorical(train_df['label'])
    train_df['label'] = train_df['label'].cat.codes

    print "Labels encoded"

    # Split data into test and train
    X_train, X_test, y_train, y_test = train_test_split(
        embedding_train[0:len(train_df)],
        train_df['label'],
        train_size=0.95,
        random_state=1234)

    print "Data split into test and train"

    # Create a LR model and train
    log_model = LogisticRegression(solver='saga', C=0.21544346900318834)
    log_model = log_model.fit(X=X_train, y=y_train)

    print "Model trained"

    # Predict
    y_pred = log_model.predict(X_test)

    print(classification_report(y_test, y_pred))

    # Create pickle of LR model
    if os.path.isfile('LRpickle.pkl'):
        os.remove('LRpickle.pkl')
    fileObject = open('LRpickle.pkl', 'wb')
    pickle.dump(log_model, fileObject)
    fileObject.close()

    print "Created pickle of model"

# START OF CODE
start_time = time.time()
print("--- %s seconds ---" % (start_time))
create_model()

print("--- %s seconds ---" % (time.time() - start_time))
