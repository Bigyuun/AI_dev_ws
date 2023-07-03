from transformers import SwinConfig, AutoImageProcessor, SwinForImageClassification, TrainingArguments
from datasets import load_dataset, load_metric

import torch
import torchvision
import numpy as np

