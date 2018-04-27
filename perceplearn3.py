from classes import input_line
from classes import Perceptron
from classes import Valence
from classes import Fake


m = Perceptron()
m.initialize('train-labeled.txt')

# print("Vanilla: ")
# for i in range(10):
#     m.train()


print("Averaged: ")
for i in range(1):
    m.train_averaged()

m.average_training_data()
m.dump_data()