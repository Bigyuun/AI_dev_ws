import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt

dataset, info = tfds.load('oxford_iiit_pet', with_info=True)