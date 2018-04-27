import json
from classes import Perceptron
import math
import pickle
import codecs
from enum import Enum
import re
import sys

m = Perceptron()
filename = sys.argv[1]
dataname = sys.argv[2]


file = open(filename)
datafile = open(dataname)
json_string = file.read()

model_type = json.loads(json_string)[0]

if model_type == "vanilla":
    print("van")

    fake_weights = json.loads(json_string)[1]
    valence_weights = json.loads(json_string)[2]
    fake_bias = json.loads(json_string)[3]
    valence_bias = json.loads(json_string)[4]
    words = json.loads(json_string)[5]
    words_dict = json.loads(json_string)[6]

    m.load_vanilla(fake_weights, valence_weights, fake_bias, valence_bias, words, words_dict)
elif model_type == "averaged":
    print("avg")

    fake_weights = json.loads(json_string)[1]
    valence_weights = json.loads(json_string)[2]
    fake_bias = json.loads(json_string)[3]
    valence_bias = json.loads(json_string)[4]
    words = json.loads(json_string)[5]
    words_dict = json.loads(json_string)[6]

    m.load_averaged(fake_weights, valence_weights, fake_bias, valence_bias, words, words_dict)
else:
    print("not a type of model")

