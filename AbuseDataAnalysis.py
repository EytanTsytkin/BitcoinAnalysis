import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


SATOSHI = 10**-8
PLOTS_PATH = '/mnt/plots/'
ABUSE_PATH = '/mnt/abuse_data/abuse.csv'

data = pd.read_csv(ABUSE_PATH)