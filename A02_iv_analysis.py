
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

# import data
prov_long = ["Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland and Labrador", "Nova Scotia", "Ontario", "Prince Edward Island", "Quebec", "Saskatchewan"]

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

        # drop first nlags rows
        full_df = full_df.dropna()
        # full_df.head()
        
        
        prods = prod_data.columns[2:16]

        result = pd.DataFrame(columns = ["industry", "param", "pval(param)", "pval(overid)", "pval(endog)", "instr"])
        # * overid: when there are more ivs than treatments, we test whether at least one of the ivs is endogenous (i.e., directly affect both T and Y) - Woolridge test
        # * endog: Test weather the treatment variable is in fact endogenous. - Wu-Hausman test

        for i in range(len(prods)):
            # i = 1
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

            # example_df.head(5)

            import statsmodels.formula.api as smf
            from linearmodels.iv import IV2SLS, IVGMMCUE

            # treatment = "tmax" # or tavg
            instruments = ["lat", "long"]
            covariates = eff_modifiers + ["log_population", "year", "month", "spring", "summer", "fall"]


            # Perform IV analyses

            iv_both = IV2SLS(
                dependent = example_df["log_production"],
                exog = example_df.loc[:, covariates],
                endog = example_df.loc[:, treatment],
                instruments = example_df.loc[:, ["lat", "long"]],
            ).fit()

            # - Overidentification test
            overid_pval = iv_both.wooldridge_overid.pval
            bonferroni_pval = 0.05/len(prods)
            

            if (overid_pval > bonferroni_pval):
                
                iv_model = iv_both

                result.loc[i] = [example_prod, 
                    np.round(iv_model.params[treatment], 3), 
                    np.round(iv_model.pvalues[treatment], 3),
                    np.round(iv_model.wooldridge_overid.pval, 3),
                    np.round(iv_model.wu_hausman().pval, 3),
                    "lat_long"]
            else:


                iv_ols = IV2SLS(
                    dependent = example_df["log_production"],
                    exog = example_df.loc[:, covariates + [treatment]],
                    endog = None,
                    instruments = None
                ).fit()

                iv_lat = IV2SLS(
                    dependent = example_df["log_production"],
                    exog = example_df.loc[:, covariates + ["long"]],
                    endog = example_df.loc[:, treatment],
                    instruments = example_df.loc[:, ["lat"]]
                ).fit()

                iv_long = IV2SLS(
                    dependent = example_df["log_production"],
                    exog = example_df.loc[:, covariates + ["lat"]],
                    endog = example_df.loc[:, treatment],
                    instruments = example_df.loc[:, ["long"]]
                ).fit()    

                
                # compare two models' coefficients to OLS: iv_lat and iv_long
                lat_diff = np.abs(iv_ols.params[treatment] - iv_lat.params[treatment])
                long_diff = np.abs(iv_ols.params[treatment] - iv_long.params[treatment])

                is_lat_endog = np.where(lat_diff < long_diff, True, False)
                
                # - If "lat" seems to be endogenous, use iv_long. Otherwise, use iv_lat.
                if (is_lat_endog):
                    iv_model = iv_long
                    iv_instr = "long"
                else:
                    iv_model = iv_lat
                    iv_instr = "lat"
                
                result.loc[i] = [example_prod, 
                    np.round(iv_model.params[treatment], 3), 
                    np.round(iv_model.pvalues[treatment], 3),
                    np.round(iv_both.wooldridge_overid.pval, 3),
                    np.round(iv_model.wu_hausman().pval, 3),
                    iv_instr]
            
        # result
        result.loc[:, "industry"] = result.loc[:, "industry"].str.replace("production_in_division_", "")
        # print(result.to_markdown())

        result_filename = "./data/user_data/01_iv_analysis/" + prov_short[i_prov] + "/" + treatment + "_log_iv2sls_result.csv"
        
        dominant_naics_filename = "./data/user_data/01_iv_analysis/" + prov_short[i_prov] + "/" + treatment + "_log_dominant_naics.csv"


        result.to_csv(result_filename, index = False)

        dominant_naics = pd.DataFrame(columns = ["GeoUID", "Dominant_NAICS"])
        d = 0
        for guid in prod_data.GeoUID.unique():
            # print(guid)
            # guid = prod_data.GeoUID.unique()[0]
            temp_df = prod_data.loc[prod_data.GeoUID == guid, :].reset_index(drop = True)
            
            if np.isnan(temp_df.loc[0, treatment]):
                continue
            else:
                dominant_naics.loc[d, "GeoUID"] = guid
                dominant_naics.loc[d, "Dominant_NAICS"] = temp_df.loc[0, "Dominant_NAICS"]
                d += 1
        
        dominant_naics2 = dominant_naics.dropna().reset_index(drop = True).value_counts("Dominant_NAICS").reset_index()
        dominant_naics2.columns = ["industry", "count"]
        # dominant_naics2

        # left join result to dominant_naics
        dominant_naics_result = pd.merge(dominant_naics2, result, left_on = "industry", right_on = "industry", how = "left")
        # dominant_naics_result
        
        dominant_naics_result.to_csv(dominant_naics_filename, index = False)

print("done!")

len(prods)