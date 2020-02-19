from model.doer import Coextractor
from subprocess import call
from os import system
import argparse
import numpy as np
import tensorflow as tf

class Main():
    def __init__(self):
        tf.logging.set_verbosity(tf.logging.ERROR)
        self.input_filename = 'res/input.txt'
        self.output_filename = 'res/output.txt'
