import random
from itertools import compress
import matplotlib.pyplot as plt
import ipdb
from seirsplus.models import *
from seirsplus.FARZ import *
import networkx
import numpy as np
import scipy.stats as stats
import time
import pickle
from os import listdir
from os.path import isfile, join
import pandas as pd

keys_saved = []
dir_results = '/mnt/DataVolume/maria/results_new/redone/'
dir_results_old = '/mnt/DataVolume/maria/results_new/'
onlyfiles = [f for f in listdir(dir_results) if isfile(join(dir_results, f))]
onlyfiles.sort()
count = 0

# first pass to get data columns
for file in onlyfiles:
    pickle_in = open(dir_results + file, "rb")
    pickle_old = open(dir_results_old + file, "rb")
    all_results_dict = pickle.load(pickle_in)
    all_results_dict_old = pickle.load(pickle_old)
    #ipdb.set_trace()
    to_save = {
        **all_results_dict_old["stats"],
        **all_results_dict["stats_unequal"],
        **all_results_dict["config_dict"]
    }
    keys_saved = list(to_save.keys() | keys_saved)
keys_saved.sort()

# second pass to fill a dataframe
df = pd.DataFrame(columns=keys_saved)
for file in onlyfiles:
    pickle_in = open(dir_results + file, "rb")
    all_results_dict = pickle.load(pickle_in)
    pickle_old = open(dir_results_old + file, "rb")
    all_results_dict_old = pickle.load(pickle_old)
    to_save = {
        **all_results_dict_old["stats"],
        **all_results_dict["stats_unequal"],
        **all_results_dict["config_dict"]
    }
    df = df.append(to_save, ignore_index=True)


df.to_csv('./postprocessed_results_unequal.csv')

