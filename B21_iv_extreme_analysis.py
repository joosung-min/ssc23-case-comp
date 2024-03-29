import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize

from collections.abc import Mapping
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS, IVGMMCUE

import os
import multiprocessing as mp

weather_prod_final = pd.read_csv("./data/user_data/02_iv_extreme/weather_prod_final.csv")
treatments = ["tmax_flag", "tmin_flag"]
prov_names = weather_prod_final["provincename"].unique().tolist()
industry_names = weather_prod_final.columns[weather_prod_final.columns.str.startswith("X")].tolist()

# Prepare covariates
covar_df = pd.read_csv("./data/user_data/_covariates/ppi_and_usd_imputed.csv")

# - min max scale each column of covar_df
from sklearn.preprocessing import MinMaxScaler
covar_sub = covar_df.drop(columns = ["date", "year", "month"])
covar_scaled = pd.DataFrame(MinMaxScaler().fit_transform(covar_sub), columns = covar_sub.columns)

# - apply PCA to covar_scaled
n_pca = 5
pca = PCA(n_components = n_pca)
covar_pca = pca.fit_transform(covar_scaled)
covar_pca = pd.DataFrame(covar_pca, columns = ["cPC"+str(i) for i in range(1, n_pca+1)])

# - add date column to covar_pca
covar_df = pd.concat([covar_df.loc[:, "date"], covar_pca], axis = 1)
covar_df["year"] = pd.to_datetime(covar_df["date"]).dt.year

# - compute yearly averages
covar_df = covar_df.groupby("year").mean().reset_index()


# Initialize result dataframe
r = 0
iv_extreme_result = pd.DataFrame(columns = ["industry", "param", "pval(param)", "pval(overid)", "pval(endog)", "provincename", "treatment", "instr"])

for treatment in treatments:
    
    for prov_name in prov_names:
        
        # - filter by province
        weather_prod = weather_prod_final.loc[weather_prod_final["provincename"] == prov_name, :].reset_index(drop = True)
        
        for industry_name in industry_names:    
            
            weather_prod["log_production"] = np.log(weather_prod[industry_name] + 1)
            weather_prod["log_population"] = np.log(weather_prod["Population"] + 1)
            
            # - merge covar_pca to weather_prod by year
            weather_prod_df = pd.merge(weather_prod, covar_df, on = "year", how = "left")

            covar_cols = ["log_population", "cPC1", "cPC2", "cPC3", "cPC4", "cPC5", "year"]
            instruments = ["lat", "lon"]
            outcome = "log_production"

            # ----------------------------
            # IV analysis
            # ----------------------------

            # - Instrumental variable analysis
            #  * IV: lat, lon
            #  * endog: log_production
            #  * exog: covariates
            example_df = weather_prod_df.copy()
            covariates = covar_cols.copy()
            
            iv_model = IV2SLS(
                dependent = example_df["log_production"],
                exog = example_df.loc[:, covariates],
                endog = example_df.loc[:, treatment],
                instruments = example_df.loc[:, ["lat", "lon"]],
            ).fit()
            
            
            temp_extreme_result = [industry_name, 
                np.round(iv_model.params[treatment], 3), 
                np.round(iv_model.pvalues[treatment], 3),
                np.round(iv_model.wooldridge_overid.pval, 3),
                np.round(iv_model.wu_hausman().pval, 3),
                prov_name,
                treatment,
                "lat_long"]
            
            iv_extreme_result.loc[r] = temp_extreme_result
            r += 1

# - is_sig == True if all of pval(param), pval(overid), pval(endog) < bonferroni_pval
bonferroni_pval = 0.05/14
iv_extreme_result["is_sig"] = np.where((iv_extreme_result["pval(param)"] < bonferroni_pval) & (iv_extreme_result["pval(overid))"] < bonferroni_pval) & (iv_extreme_result["pval(endog)"] < bonferroni_pval), True, False)
iv_extreme_result["new_param"] = np.where(iv_extreme_result["is_sig"] == True, iv_extreme_result["param"], 0)

iv_extreme_result.to_csv("./data/user_data/02_iv_extreme/iv_extreme_result.csv", index = False)

iv_extreme_result.loc[iv_extreme_result["is_sig"] == True, :]