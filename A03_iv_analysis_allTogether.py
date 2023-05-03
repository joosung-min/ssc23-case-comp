
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

prov_short = ["AB", "BC", "MB", "NB", "NL", "NS", "ON", "PE", "QC", "SK"]

prod_full = pd.DataFrame()
for i_prov in range(len(prov_short)):
    # i_prov = 1

    prod_filename = "./data/user_data/01_iv_analysis/" + prov_short[i_prov] + "/prod_temp.csv"

    prod_data = pd.read_csv(prod_filename)
    prod_full = pd.concat([prod_full, prod_data], axis = 0)


prod_full = prod_full.dropna().reset_index(drop = True)
prod_full = prod_full.rename(columns = {"Date":"date"})
prod_full["tdiff"] = np.abs(prod_full.tmax - prod_full.tmin)
prod_full["log_population"] = np.log(prod_full.Population + 1)

# - load covariates
covar_df = pd.read_csv("./data/user_data/_covariates/ppi_and_usd_imputed.csv")
covar_df.head()

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

# - add date column to covar_pca
covar_df = pd.concat([covar_df.loc[:, "date"], covar_pca], axis = 1)
# covar_df.head()

# - left join the covariates to the prod_data by date.
full_df_raw = pd.merge(prod_full, covar_df, on='date', how='left')

# - add lagged covariates
ppi_covars = covar_pca.columns.tolist()
lags = 3
for lag in range(1, lags+1):
    for covar in ppi_covars:
        full_df_raw[covar + "_lag_" + str(lag)] = full_df_raw[covar].shift(lag)

full_df_raw = full_df_raw.dropna()

# - Create dummy variables for provincename
full_df = pd.concat([full_df_raw.drop(columns = ["provincename"]), pd.get_dummies(full_df_raw.provincename, columns = ["provincename"], drop_first=True)], axis = 1)

# - Replace " " with  "_" in all column names
full_df.columns = full_df.columns.str.replace(" ", "_")

binary_provs = full_df.columns[-9:].tolist()
# binary_provs

# - extract columns that starts with "production_in_division"
prod_cols = [col for col in full_df.columns if col.startswith("production_in_division")]

treatments = ["tavg", "tmax", "tmin", "tdiff"]
for treatment in treatments:
    # treatment = "tavg"

    result = pd.DataFrame(columns = ["industry", "param", "pval(param)", "pval(overid)", "pval(endog)"])
    
    for i in range(len(prod_cols)):
        # i = 9
        example_prod = prod_cols[i]
        # example_prod
        temperatures = ["tavg", "tmin", "tmax", "tdiff"]
        instruments = ["lat", "long"]

        eff_modifiers = [s + "_lag_"+str(lags) for s in ppi_covars]
        # eff_modifiers.append("Population")
        eff_modifiers.append("log_population")
        eff_modifiers = eff_modifiers + binary_provs
        # eff_modifiers
        com_causes = ["year", "month"]

        example_cols = [example_prod] + eff_modifiers + com_causes + temperatures + instruments + ["date"]
        example_df = full_df.loc[:, example_cols].reset_index(drop= True)
        example_df = example_df.rename(columns = {example_prod:"production"})

        # create log(production) column. First add 1 to all values of production to avoid taking log of 0.
        example_df.loc[:, "log_production"] = np.log(example_df.production + 1)

        # create columns for the seasons of the year
        example_df.loc[:, "spring"] = (example_df.month.isin([3,4,5])).astype(int)
        example_df.loc[:, "summer"] = (example_df.month.isin([6,7,8])).astype(int)
        example_df.loc[:, "fall"] = (example_df.month.isin([9,10,11])).astype(int)
        example_df.loc[:, "winter"] = (example_df.month.isin([12,1,2])).astype(int)

        # example_df.head(5)

        import statsmodels.formula.api as smf
        from linearmodels.iv import IV2SLS, IVGMMCUE

        def parse(model, exog):
            param = model.params[exog]
            se = model.std_errors[exog]
            p_val = model.pvalues[exog]
            print(f"Parameter: {param}")
            print(f"SE: {se}")
            print(f"95CI: {(-1.96*se, 1.96*se) + param}")
            print(f"P-value: {p_val}")

        instruments = ["lat", "long"]        
        
        seasons = ["spring", "summer", "fall"]
        formula = 'log_production ~ 0 + year + ' + "+".join(seasons) + "+" + ' + '.join(eff_modifiers) + \
            "+ [" + treatment + " ~ " + ' + '.join(instruments) + " ]"
    
        result_filename = "./data/user_data/01_iv_analysis/_allTogether/" + treatment + "_iv2sls_result.csv"
        dominant_naics_filename = "./data/user_data/01_iv_analysis/_allTogether/" + treatment + "_dominant_naics.csv"

        iv_model = IV2SLS.from_formula(formula, data = example_df).fit(cov_type = "robust", debiased = True)
        # iv_model = IVGMMCUE.from_formula(formula, data = example_df).fit(cov_type = "robust", debiased = False)
        # print(example_prod)
        # parse(iv_model, treatment)
        print(iv_model)
    
        # print(example_prod)
        # display(iv_model)
        result.loc[i] = [example_prod, 
            np.round(iv_model.params[treatment], 3), 
            np.round(iv_model.pvalues[treatment], 3),
            np.round(iv_model.basmann.pval, 3),
            np.round(iv_model.wooldridge_score.pval, 3)
        ]
    # result
    result.loc[:, "industry"] = result.loc[:, "industry"].str.replace("production_in_division_", "")
    # print(result.to_markdown())

    result.to_csv(result_filename, index = False)

    dominant_naics = pd.DataFrame(columns = ["GeoUID", "Dominant_NAICS"])
    d = 0
    for guid in prod_full.GeoUID.unique():
        # print(guid)
        # guid = prod_data.GeoUID.unique()[0]
        temp_df = prod_full.loc[prod_full.GeoUID == guid, :].reset_index(drop = True)
        
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