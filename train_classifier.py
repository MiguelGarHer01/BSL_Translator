import os
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

data_dict = pickle.load(open('./expressions.pkl', 'wb'))

expressions = data_dict['expressions']
labels = data_dict['labels']

