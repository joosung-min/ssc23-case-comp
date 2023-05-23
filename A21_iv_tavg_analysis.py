
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


prov_short = ["AB", "BC", "MB", "NB", "NL", "NS", "ON", "PE", "QC", "SK"]

treatments = ["tmax", "tavg", "tdiff"]

# prepare cPCs
covar_df = pd.read_csv("./data/user_data/_covariates/ppi_and_usd_imputed.csv")
# covar_df.head()

# - min max scale each column of covar_df
from sklearn.preprocessing import MinMaxScaler
covar_sub = covar_df.drop(columns = ["date", "year", "month"])
covar_scaled = pd.DataFrame(MinMaxScaler().fit_transform(covar_sub), columns = covar_sub.columns)
# covar_sub.head()

# - apply PCA to covar_scaled
n_pca = 5
pca = PCA(n_components = n_pca)
covar_pca = pca.fit_transform(covar_scaled)
covar_pca = pd.DataFrame(covar_pca, columns = ["PC"+str(i) for i in range(1, n_pca+1)])
ppi_covars = covar_pca.columns.tolist()

# - add date column to covar_pca
covar_df = pd.concat([covar_df.loc[:, "date"], covar_pca], axis = 1)


for treatment in treatments:
    # treatment = "tmax"
    for i_prov in range(len(prov_short)):
        # i_prov = 1

        prod_filename = "./data/user_data/01_iv_analysis/" + prov_short[i_prov] + "/prod_temp.csv"

        prod_data = pd.read_csv(prod_filename)
        prod_data = prod_data.rename(columns = {"Date":"date"})
        prod_data["tdiff"] = np.abs(prod_data.tmax - prod_data.tmin)
        prod_data["log_population"] = np.log(prod_data.Population + 1)

        # - left join the cPCs to the prod_data by date.
        full_df = pd.merge(prod_data, covar_df, on='date', how='left')
        
        # - apply lags to ppi_covars
        lags = 3
        for lag in range(1, lags+1):
            for covar in ppi_covars:
                full_df[covar + "_lag_" + str(lag)] = full_df[covar].shift(lag)

        full_df = full_df.dropna()
        
        prods = prod_data.columns[2:16]

        result = pd.DataFrame(columns = ["industry", "param", "pval(param)", "pval(endog)", "instr"])
        
        for i in range(len(prods)):
            
            example_prod = prods[i]
            temperatures = ["tavg", "tmin", "tmax"]
            instruments = ["lat", "long"]

            eff_modifiers = [s + "_lag_"+str(lags) for s in ppi_covars]

            example_cols = [example_prod] + eff_modifiers + temperatures + instruments + ["date", "year", "month", "log_population"]
            example_df = full_df.loc[:, example_cols].reset_index(drop= True)
            example_df = example_df.rename(columns = {example_prod:"production"})

            # - create log(production) column. 
            example_df.loc[:, "log_production"] = np.log(example_df.production + 1)
            example_df.loc[:, "tdiff"] = np.abs(example_df.tmax - example_df.tmin)

            # - create columns for the seasons of the year
            example_df.loc[:, "spring"] = (example_df.month.isin([3,4,5])).astype(int)
            example_df.loc[:, "summer"] = (example_df.month.isin([6,7,8])).astype(int)
            example_df.loc[:, "fall"] = (example_df.month.isin([9,10,11])).astype(int)
            example_df.loc[:, "winter"] = (example_df.month.isin([12,1,2])).astype(int)


            import statsmodels.formula.api as smf
            from linearmodels.iv import IV2SLS, IVGMMCUE

            instruments = ["lat", "long"]
            covariates = eff_modifiers + ["log_population", "year", "month", "spring", "summer", "fall"]


            # - Perform IV analyses
            iv_model = IV2SLS(
                dependent = example_df["log_production"],
                exog = example_df.loc[:, covariates],
                endog = example_df.loc[:, treatment],
                instruments = example_df.loc[:, ["lat", "long"]],
            ).fit()

            # * endog: Test weather the treatment variable is in fact endogenous. - Wu-Hausman test. H0: All endogenous variables are exogenous.

            result.loc[i] = [example_prod, 
                np.round(iv_model.params[treatment], 3), 
                np.round(iv_model.pvalues[treatment], 3),
                np.round(iv_model.wu_hausman().pval, 3),        # endog test
                "lat_long"]
            
    
        result.loc[:, "industry"] = result.loc[:, "industry"].str.replace("production_in_division_", "")

        result_filename = "./data/user_data/01_iv_analysis/" + prov_short[i_prov] + "/" + treatment + "_log_iv2sls_result.csv"    
        dominant_naics_filename = "./data/user_data/01_iv_analysis/" + prov_short[i_prov] + "/" + treatment + "_log_dominant_naics.csv"

        result.to_csv(result_filename, index = False)

print("done!")


result_combined = pd.DataFrame()

for prov in prov_short:
    
    temp_result = pd.read_csv("./data/user_data/01_iv_analysis/" + prov + "/tmax_log_iv2sls_result.csv")
    temp_result["is_sig"] = np.where((temp_result["pval(param)"] < bonferroni_pval) & (temp_result["pval(endog)"] < bonferroni_pval), True, False)
    temp_result["new_param"] = np.where(temp_result["is_sig"] == True, temp_result["param"], 0)
    temp_result["provincename"] = prov
    
    result_combined = pd.concat([result_combined, temp_result], axis = 0)

result_combined.to_csv("./data/user_data/01_iv_analysis/A02_iv_tavg_result.csv", index = False)