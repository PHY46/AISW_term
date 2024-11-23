# %%
import pandas as pd
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.datasets import TranslationDataset, Multi30k
from torchtext.legacy.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time

# %%
df = pd.DataFrame(columns = ['원문','번역문'])
path = './data/'

file_list = [ '2_대화체.xlsx',
 '1_구어체(2).xlsx',
 '1_구어체(1).xlsx',
 '3_문어체_뉴스(2).xlsx',
 '3_문어체_뉴스(3).xlsx',
 '4_문어체_한국문화.xlsx']

for data in file_list:
    temp = pd.read_excel(path+data)
    df = pd.concat([df,temp[['원문','번역문']]])
# %%
df.head()
# %%
