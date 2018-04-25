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
        return self.valence_assigned == self.valence_true

    def correct_fake(self):
        return self.fake_assigned == self.fake_true

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

    def __init__(self, input):
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

        # initialize bias
        self.bias_fake = 0
        self.bias_valence = 0



    def train(self, input):
        for line in self.lines:



        with open('nbmodel.txt', 'w') as fp:
            json.dump((self.P_fake_fake, self.P_fake_real, self.P_val_neg, self.P_val_pos,
            self.prior_real, self.prior_fake, self.prior_pos, self.prior_neg), fp)

    def classify_valence(self, ):


    def classify_fake(self, ):

    def extract_line(self, line):
        temp_line = line.split(' ')
        temp_input_line = input_line(temp_line[0], temp_line[3:], temp_line[2], temp_line[1])
        self.lines_saved.append(temp_input_line)
        for w in temp_input_line.words:
            regex = re.compile(r"[^a-zA-Z]")
            word = re.sub(regex, '', w).lower()
            try:
                self.stops[word] += 1
            except KeyError:
                # this means the word was not a stop word, try and put it
                try:
                    self.words_dict[word] += 1
                except KeyError:
                    self.words_dict[word] = 1
                if temp_input_line.valence_true is Valence.neg:
                    self.num_val_neg += 1
                    try:
                        self.count_val_neg[word] += 1
                    except KeyError:
                        self.count_val_neg[word] = 1
                if temp_input_line.valence_true is Valence.pos:
                    self.num_val_pos += 1
                    try:
                        self.count_val_pos[word] += 1
                    except KeyError:
                        self.count_val_pos[word] = 1
                if temp_input_line.fake_true is Fake.fake:
                    self.num_fake_fake += 1
                    try:
                        self.count_fake_fake[word] += 1
                    except KeyError:
                        self.count_fake_fake[word] = 1
                if temp_input_line.fake_true is Fake.real:
                    self.num_fake_real += 1
                    try:
                        self.count_fake_real[word] += 1
                    except KeyError:
                        self.count_fake_real[word] = 1
