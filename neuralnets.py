import re
from nltk.corpus import stopwords
import pandas as pd
import time
import numpy as np
from numpy import array
import torch
from torch import autograd
import torch.nn.functional as F
import torch.nn as nn
import pickle

bag_of_words = set()
stop_words = set(stopwords.words('english'))


# Remove non ASCII words
def remove_non_ascii(words):
    return "".join(i for i in words if ord(i) < 128)


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

    train_df['label'] = pd.Categorical(train_df['label'])
    train_df['label'] = train_df['label'].cat.codes

    test_df['label'] = pd.Categorical(test_df['label'])
    test_df['label'] = test_df['label'].cat.codes

    print "Labels encoded"

    return train_df, test_df


# Class to define NN architecture
class Net(nn.Module):
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(input_neurons, hidden_neurons)
        self.layer_2 = nn.Linear(hidden_neurons, output_neurons)

    def forward(self, x):
        out = self.layer_1(x)
        out = F.tanh(out)
        out = self.layer_2(out)
        return out


# Training NN
def neural_net(train_df):
    input_neurons=len(bag_of_words)
    hidden_neurons=5
    output_neurons=1
    learning_r=0.01
    net = Net(input_neurons, hidden_neurons, output_neurons)
    opt=torch.optim.Adam(params=net.parameters(),lr=learning_r)
    
    label_train = np.array(train_df['label'])
    train_review_size = len(label_train)
    label_train = array(label_train).reshape(train_review_size, 1)

    for epoch in range(4):
        for i, row in train_df.iterrows():
	    print i
            embedding_train = [0] * len(bag_of_words)
            for word in row['text'].split():
                if word in bag_of_words:
                    embedding_train[word2int[word]] = 1
            input_num=torch.from_numpy(np.array(embedding_train,dtype=np.float32))
            inputs=autograd.Variable(input_num)
            target_num=torch.from_numpy(np.array(label_train[i],dtype=np.float32))
            target=autograd.Variable((target_num))
            output=net(inputs)
            loss = nn.MSELoss()
            loss_is=loss(output,target)
            opt.zero_grad()
            loss_is.backward()
            opt.step()
    return net


# START OF CODE

train_df, test_df = feature_extraction()

start_time = time.time()
print("--- %s seconds ---" % (start_time))

word2int = {}
for i, word in enumerate(bag_of_words):
    word2int[word] = i


net=neural_net(train_df)
filename = 'finalized_model.sav'
pickle.dump(net, open(filename, 'wb'))
output=open("result.txt","w")

# Validation
label_test = np.array(test_df['label'])
test_review_size = len(label_test)
label_test = array(label_test).reshape(test_review_size, 1)

for i, row in test_df.iterrows():
    embedding_test = [0] * len(bag_of_words)
    for word in row['text'].split():
        if word in bag_of_words:
            embedding_test[word2int[word]] = 1
    input_num=torch.from_numpy(np.array(embedding_test,dtype=np.float32))
    input=autograd.Variable(input_num)
    output.write(str(net(input)) +"\t")
    output.write(str(label_test[i]) + "\n")


print("--- %s seconds ---" % (time.time() - start_time))