# import modules

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# loading data

data = tf.keras.datasets.cifar10

(train_ , train_lbl) , (test_ , test_lbl) = data.load_data()