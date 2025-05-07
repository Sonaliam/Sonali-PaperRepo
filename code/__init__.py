import json
import pickle
import numpy as np
import sys
import os
import nltk
import spacy
import math
import time
import argparse
from nltk.translate.bleu_score import corpus_bleu
import os.path
from pycocotools.coco import COCO
from collections import Counter
import tensorflow as tf
import scipy.misc

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from PIL import Image
from tqdm import tqdm
import random
import json

from transformers import CLIPModel, CLIPProcessor, RobertaModel, RobertaTokenizer
from facenet_pytorch import MTCNN, InceptionResnetV1
import open_clip
from torchvision.transforms import ToPILImage