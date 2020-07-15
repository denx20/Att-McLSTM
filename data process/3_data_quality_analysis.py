from math import *
import numpy as np
import os
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from openpyxl.reader.excel import load_workbook
from openpyxl import Workbook
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras import regularizers
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import *
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Dense, Input, concatenate, LSTM, merge
from keras.layers.core import *
import random
import time

path = 'C:\\Users\\HP\\Desktop\\实验\\流感疫情分析\\工作进展\\2_LSTM预测\\4_lstm_for_structure2\\content'
files= os.listdir(path)
num = 9
print(files[num])
filename = path + '\\' + files[num]
wb = load_workbook(filename = filename)
ws = wb.active 
temp1 = []
for row in ws.rows:
    temp2 = []
    for col in row:
        temp2.append(col.value)
    temp1.append(temp2)
temp2 = temp1[1:]

content = temp2
vir = []
con = 11
for i in content:
    vir.append(i[con])
plt.plot(vir)
