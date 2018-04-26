from classes import input_line
from classes import Perceptron
from classes import Valence
from classes import Fake


m = Perceptron('train-labeled.txt')

for i in range(10):
    m.train()
