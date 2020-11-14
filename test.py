import random
import numpy as np
import tensorflow as tf
from konlpy.tag import Okt

from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Lambda, Layer, Embedding, LayerNormalization

import os
os.chdir('D:\OneDrive\대학원\딥러닝\패스트캠퍼스 강의\Part5) 딥러닝 최신 트렌드\자연어처리 분야 딥러닝 모델 ( Natural Language Processing )\data')

EPOCHS = 200
NUM_WORDS = 2000

class DotScaledAttention(Layer):
    pass