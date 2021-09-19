import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
import os

# nltk.download('punkt')
steamer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

if os.path.exists("data.pickle"):
    with open("data.pickle", "rb") as file:
        words, labels, training, output = pickle.load(file)
else:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            docs_x.append(word_list)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [steamer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in labels]

    for x, doc in enumerate(docs_x):
        bag = []

        word_list = [steamer.stem(w) for w in doc]

        for w in words:
            if w in word_list:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as file:
        pickle.dump((words, labels, training, output), file)

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
if os.path.exists("model.tflearn.meta"):
    model.load("model.tflearn")
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(sentence, words_list):
    bag_words = [0 for _ in words_list]

    s_words = nltk.word_tokenize(sentence)
    s_words = [steamer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words_list):
            if w == se:
                bag_words[i] = 1

    return np.array(bag_words)


def chat():
    print("Start talking")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        result = model.predict([bag_of_words(inp, words)])[0]
        result_index = np.argmax(result)

        if result[result_index] > 0.8:
            label = labels[result_index]

            responses = []
            for tag in data["intents"]:
                if tag["tag"] == label:
                    responses = tag["responses"]
            print(random.choice(responses))
        else:
            print("I don't quite understand. Please ask a different question")


chat()
