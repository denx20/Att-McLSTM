from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from openpyxl.reader.excel import load_workbook
from openpyxl import Workbook
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import random

#constant
total_week = 468

filename = ''
wb = load_workbook(filename=filename)
ws = wb.active 
temp = []
temp1 = []
for row in ws.rows:
    temp2 = []
    for col in row:
        temp2.append(col.value)
    temp1.append(temp2)
content = temp1[1: total_week + 1]

vir = []
con = 21
for i in content:
    vir.append(i[con])
plt.plot(vir)