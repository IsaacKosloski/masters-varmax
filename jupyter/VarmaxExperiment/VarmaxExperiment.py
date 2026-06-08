import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VARMAX
from statsmodels.iolib.smpickle import load_pickle
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from itertools import product,combinations
from pathlib import Path

class VarmaxExperiment:
    def __init__(self):
        pass

    def model_gridsearch_dataset(self,file_path:Path):