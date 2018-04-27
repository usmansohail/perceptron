import numpy as np
import json
import math
import pickle
import codecs
from enum import Enum
import re
import sys


class Valence(Enum):
    pos = 1
    neg = -1


class Fake(Enum):
    real = 1
    fake = -1

class input_line():
    valence_assigned = None
    valence_true = None
    fake_assigned = None
    fake_true = None
    key = None
    words = []

    def __init__(self, key, line, val, fake):
        self.key = key
        self.words = line
        self.add_fake_true(fake)
        self.add_valence_true(val)

    def add_valence_true(self, val):
        if val == 'Pos':
            self.valence_true = Valence.pos
        elif val == 'Neg':
            self.valence_true = Valence.neg
        else:
            print("Something wrong with Valence of the line: ",
                  '\n', val, '\n')

    def add_fake_true(self, fake_true):
        if fake_true == 'Fake':
            self.fake_true = Fake.fake
        elif fake_true == "True":
            self.fake_true = Fake.real
        else:
            print("Something wrong with fake: ",
                  '\n', fake_true, '\n')

    def add_valence_assigned(self, Valence):
        self.valence_assigned = Valence

    def add_fake_assigned(self, fake):
        self.fake_assigned = fake

    def correct_valence(self):
        return self.valence_true.value

    def correct_fake(self):
        return self.fake_true.value

    def get_vector(self, size, words_dict):
        vector = [0] * size;

        for word in self.words:
            vector[words_dict[word]] += 1

        vector = np.array([vector])

        return vector



class Perceptron():
    weights_valence = None
    weights_fake = None
    bias_valence = None
    bias_fake = None
    lines = []
    words_dict = {}
    words = []
    dimension = None
    count = None
    cache_valence = None
    cache_fake = None
    cache_bias_valence = None
    cache_bias_fake = None
    averaged_valence_weights = None
    averaged_fake_weights = None
    averaged_valence_bias = None
    averaged_fake_bias = None

    def __init__(self):
        pass

    def initialize(self, input):
        # read all words in the training file, store the lines
        file = codecs.open(input, 'rb', 'utf-8')
        for line in file.readlines():
            temp_line = line.split(' ')
            temp_input_line = input_line(temp_line[0], temp_line[3:], temp_line[2], temp_line[1])
            self.lines.append(temp_input_line)

            for word in temp_input_line.words:
                self.words_dict[word] = 1

        # get the list of words
        self.words = dict(self.words_dict).keys()

        # store the index of the word in the dictionary
        for i, word in enumerate(self.words):
            self.words_dict[word] = i


        # initialize an array to 0 of size dimensions
        self.dimension = len(dict(self.words_dict).keys())
        self.weights_fake = [0] * self.dimension
        self.weights_valence = [0] * self.dimension

        self.weights_fake = np.array([self.weights_fake])
        self.weights_valence = np.array([self.weights_valence])

        # initialize bias
        self.bias_fake = 0
        self.bias_valence = 0

        # variables for averaged perceptron
        self.averaged_valence_weights = np.array([[0] * self.dimension])
        self.averaged_fake_weights = np.array([[0] * self.dimension])
        self.averaged_valence_bias = 0
        self.averaged_fake_bias = 0



    def train(self):
        incorrect_valence =  0
        incorrect_fake =  0

        for line in self.lines:
            valence_result = self.classify_valence(line.get_vector(self.dimension, self.words_dict))
            fake_result = self.classify_fake(line.get_vector(self.dimension, self.words_dict))

            if line.correct_valence() * valence_result <= 0:
                add_val = line.correct_valence() * line.get_vector(self.dimension, self.words_dict)
                self.weights_valence += add_val
                self.bias_valence += line.correct_valence()
                incorrect_valence += 1


            if line.correct_fake() * fake_result <= 0:
                self.weights_fake += line.correct_fake() * line.get_vector(self.dimension, self.words_dict)
                self.bias_fake += line.correct_fake()
                incorrect_fake += 1

        # display results
        print("Incorrect fake: ", incorrect_fake, "Incorrect valence: ", incorrect_valence)
        return (incorrect_fake, incorrect_valence)


    def train_averaged(self):
        incorrect_valence = 0
        incorrect_fake = 0

        # reset values every training session
        self.count = 0
        self.cache_valence = np.array([[0] * self.dimension])
        self.cache_fake = np.array([[0] * self.dimension])
        self.cache_bias_valence = 0
        self.cache_bias_fake = 0

        for line in self.lines:
            valence_result = self.classify_averaged_valence(line.get_vector(self.dimension, self.words_dict))
            fake_result = self.classify_averaged_fake(line.get_vector(self.dimension, self.words_dict))
            self.count += 1

            if line.correct_valence() * valence_result <= 0:
                add_val = line.correct_valence() * line.get_vector(self.dimension, self.words_dict)
                self.averaged_valence_weights += add_val
                self.averaged_valence_bias += line.correct_valence()
                incorrect_valence += 1

                self.cache_valence += (line.correct_valence() * self.count) * \
                    line.get_vector(self.dimension, self.words_dict)
                self.cache_bias_valence += line.correct_valence() * self.count

            if line.correct_fake() * fake_result <= 0:
                self.averaged_fake_weights += line.correct_fake() * line.get_vector(self.dimension, self.words_dict)
                self.averaged_fake_bias += line.correct_fake()
                incorrect_fake += 1

                self.cache_fake += line.correct_fake() * \
                    line.get_vector(self.dimension, self.words_dict) * self.count
                self.cache_bias_fake += line.correct_fake() * self.count

        # display results
        print("Incorrect fake: ", incorrect_fake, "Incorrect valence: ", incorrect_valence)
        return (incorrect_fake, incorrect_valence)

    def average_training_data(self):
        # average the weights
        self.averaged_valence_weights = self.averaged_valence_weights - (float(1 / self.count) * self.cache_valence)
        self.averaged_fake_weights = self.averaged_fake_weights - (float(1 / self.count) * self.cache_fake)

        self.averaged_valence_bias = self.bias_valence - (float(1 / self.count) * self.cache_bias_valence)
        self.averaged_fake_bias = self.bias_fake - (float(1 / self.count) * self.cache_bias_fake)



    def classify_valence(self, x):
        return np.dot(x, np.transpose(self.weights_valence)) + self.bias_valence


    def classify_fake(self, x):
        return np.dot(x, np.transpose(self.weights_fake)) + self.bias_fake


    def classify_averaged_valence(self, x):
        return np.dot(x, np.transpose(self.averaged_valence_weights)) + self.averaged_valence_bias

    def classify_averaged_fake(self, x):
        return np.dot(x, np.transpose(self.averaged_fake_weights)) + self.averaged_fake_bias

    def dump_data(self):
        with open('vanillamodel.txt', 'w') as fp:
            json.dump(("vanilla", self.weights_fake.tolist(), self.weights_valence.tolist(), self.bias_fake,
                       self.bias_valence, list(self.words), self.words_dict), fp)
        with open('averagedmodel.txt', 'w') as fp:
            json.dump(("averaged", self.averaged_fake_weights.tolist(), self.averaged_valence_weights.tolist(),
                       self.averaged_fake_bias, self.averaged_valence_bias, list(self.words), self.words_dict), fp)


    def load_vanilla(self, weights_fake, weights_valence, bias_fake, bias_valence, words, words_dict):
        self.weights_fake = weights_fake
        self.weights_valence = weights_valence
        self.bias_fake = bias_fake
        self.bias_valence = bias_valence
        self.words = words
        self.words_dict = words_dict


    def load_averaged(self, averaged_fake_weights, averaged_valence_weights, averaged_fake_bias,
                      averaged_valence_bias, words, words_dict):
        self.averaged_fake_weights = averaged_fake_weights
        self.averaged_valence_weights = averaged_valence_weights
        self.averaged_fake_bias = averaged_fake_bias
        self.averaged_valence_bias = averaged_valence_bias
        self.words = words
        self.words_dict = words_dict


