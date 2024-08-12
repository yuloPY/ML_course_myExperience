# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header=None)
L = []
for i in range(0,len(dataset)):
    L.append([str(dataset.values[i,j]) for j in range(0,20)])

# Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions=L,min_support=(3*7)/len(dataset),min_confidence=0.2,min_lift=3,min_length=2,max_lenght=2)

