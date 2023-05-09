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

os.chdir("/home/joosungm/projects/def-lelliott/joosungm/projects/ssc23-case-comp")

# Import data
weather_prod_final = pd.read_csv("./data/user_data/02_counterfactual_analysis/weather_prod_final.csv")
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
iv_extreme_result = pd.DataFrame(columns = ["industry", "param", "pval(param)", "pval(overid)", "pval(endog)", "provincename", "treatment"])

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
            # Check linear relationships
            # ----------------------------
            # - Check linear relationships
            # smf.ols(formula = treatment + " ~ " + "+".join(instruments), data = weather_prod_df).fit().summary()
            # smf.ols(formula = "log_production ~ tmax_flag", data = weather_prod_df).fit().summary()


            # ----------------------------
            # IV analysis
            # ----------------------------

            # - Instrumental variable analysis
            #  * IV: lat, lon
            #  * endog: log_production
            #  * exog: covariates

            formula = outcome + " ~ 0 +" + "+".join(covar_cols) + "+[" + treatment + "~" + "+".join(instruments) + "]"
            iv_model = IV2SLS.from_formula(formula, data = weather_prod_df).fit(cov_type = "robust", debiased = True)

            # - collect results
            temp_extreme_result = [industry_name, 
                                np.round(iv_model.params[treatment], 3),
                                np.round(iv_model.pvalues[treatment], 3),
                                np.round(iv_model.basmann.pval, 3),
                                np.round(iv_model.wooldridge_score.pval, 3),
                                prov_name,
                                treatment
                                ]
            iv_extreme_result.loc[r] = temp_extreme_result
            r += 1

iv_extreme_result
bonferroni_pval = 0.05/len(industry_names)
# - is_sig == True if all of pval(param), pval(overid), pval(endog) < bonferroni_pval
iv_extreme_result["is_sig"] = np.where((iv_extreme_result["pval(param)"] < bonferroni_pval) & (iv_extreme_result["pval(overid)"] < bonferroni_pval) & (iv_extreme_result["pval(endog)"] < bonferroni_pval), True, False)
iv_extreme_result["new_param"] = np.where(iv_extreme_result["is_sig"] == True, iv_extreme_result["param"], 0)

iv_extreme_result.to_csv("./data/user_data/02_iv_extreme/iv_extreme_result.csv", index = False)

iv_extreme_result.loc[iv_extreme_result["is_sig"] == True, :]