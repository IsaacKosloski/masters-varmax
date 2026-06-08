import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VARMAX
from statsmodels.iolib.smpickle import load_pickle
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from itertools import product,combinations
from pathlib import Path

class VarmaxExperiment:
    def __init__(self,data,exog_cols,endog_cols,p,q,x_lag,real_degree,img_degree) -> None:
        self.data:pd.DataFrame = data
        self.exog_cols:list = exog_cols
        self.endog_cols:list = endog_cols
        self.p:int = p
        self.q:int = q
        self.x_lag:int = x_lag
        self.real_degree:int = real_degree
        self.img_degree:int = img_degree
        self.model:VARMAX = None
        self.exp_data:pd.DataFrame = None
        self.exp_exog_cols:list = None

    def define_experiment_data(self):
        self.exp_data = self.data.copy()
        self.exp_exog_cols = list(self.exog_cols)

        if(self.x_lag > 0):
            self.exp_data,self.exp_exog_cols = VarmaxExperiment.select_auto_lag(self.data,self.exog_cols,max_x_lag=self.x_lag)

        self.exp_data,self.exp_exog_cols = VarmaxExperiment.select_auto_polinomial_degree(self.exp_data,self.exp_exog_cols,'r',self.real_degree)
        self.exp_data,self.exp_exog_cols = VarmaxExperiment.select_auto_polinomial_degree(self.exp_data,self.exp_exog_cols,'i',self.img_degree)

    def load_model(self,path:str) -> None:
        params = {'p':self.p,'x_lag':self.x_lag,'r':self.real_degree,'i':self.img_degree}
        model_pkl_path = Path(f"{path}")

        if model_pkl_path.exists():
            self.define_experiment_data()
            self.model = load_pickle(str(model_pkl_path))
        else:
            print("[Error] File not found!")

    @staticmethod
    def select_auto_lag(df_exog,exog_cols,max_x_lag=2,inplace=True) -> pd.DataFrame:
        """
        Aplica os lags desejado às exógenas da base de dados original e retorna o dataframe com as novas colunas
        """
        new_exog_cols = list(exog_cols)

        for lag in range(1, max_x_lag + 1):
            for col in exog_cols:
                lag_cols_name = f'{col}_lag{lag}'
                df_exog[lag_cols_name] = df_exog[col].shift(lag)
                new_exog_cols.append(lag_cols_name)

        lag_cols = [f'{col}_lag{lag}' for lag in range(1, max_x_lag+1) for col in exog_cols]
        rows_to_drop = df_exog.index[df_exog[lag_cols].isna().any(axis=1)]

        df_exog = df_exog.drop(index=rows_to_drop)
        return df_exog,new_exog_cols
    
    @staticmethod
    def select_auto_polinomial_degree(df_exog,exog_cols,degree_type,max_degree=1) -> pd.DataFrame:
        """
        Aplica os graus desejado às exógenas da base de dados original, gerando o polinômio completo para cada coluna
        das exógenas (originais e sem lags) e retorna o dataframe com as novas colunas
        """
        new_exog_cols = list(exog_cols)
        new_columns = {}

        if degree_type in ['real','r','REAL','Real']:
            degree_type = 'real'
        elif degree_type in ['img','i','IMG','Img']:
            degree_type = 'img'
        else:
            raise ValueError("degree_type must be 'real' or 'img'")

        if max_degree >= 2:
            for degree in range(2, max_degree + 1):
                columns = [col for col in exog_cols if degree_type in col]
                for col in columns:
                    degree_col_name = f'{col}_deg{degree}'
                    new_columns[degree_col_name] = df_exog[col] ** degree
                    new_exog_cols.append(degree_col_name)

        if new_columns:
            df_exog = pd.concat([df_exog,pd.DataFrame(new_columns,index=df_exog.index)],axis=1)

        return df_exog,new_exog_cols