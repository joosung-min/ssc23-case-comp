
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

scales = ["raw_", "log_"]

treatments = ["tmax", "tavg", "tdiff"]

for treatment in treatments:
    # treatment = "tmax"
    for scale in scales:
        # scale = "log_"
        for i_prov in range(len(prov_short)):
            # i_prov = 1

            prod_filename = "./data/user_data/01_iv_analysis/" + prov_short[i_prov] + "/prod_temp.csv"

            prod_data = pd.read_csv(prod_filename)
            prod_data = prod_data.rename(columns = {"Date":"date"})
            prod_data["tdiff"] = np.abs(prod_data.tmax - prod_data.tmin)
            prod_data["log_population"] = np.log(prod_data.Population + 1)
            # prod_data.head()
            
            # - load covariates
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
            
            # - add date column to covar_pca
            covar_df = pd.concat([covar_df.loc[:, "date"], covar_pca], axis = 1)
            # covar_df.head()


            # - left join the covariates to the prod_data by date.
            full_df = pd.merge(prod_data, covar_df, on='date', how='left')

            # ppi_covars = ["PC"+str(i) for i in range(1, n_pca+1)]
            ppi_covars = covar_pca.columns.tolist()
            # ppi_covars
            # apply lags to ppi_covars
            lags = 3
            for lag in range(1, lags+1):
                for covar in ppi_covars:
                    full_df[covar + "_lag_" + str(lag)] = full_df[covar].shift(lag)

            # drop first nlags rows
            full_df = full_df.dropna()
            # full_df.head()
            
            # Create example data for analysis
            # - Subset date, lat, long, the covariates, tavg, and production_in_division_X31.33.Manufacturing from full_df
            # and then rename the column to 'production'
            prods = prod_data.columns[2:16]

            result = pd.DataFrame(columns = ["industry", "param", "pval(param)", "pval(overid)", "pval(endog)"])
            for i in range(len(prods)):
                # i = 1
                example_prod = prods[i]
                temperatures = ["tavg", "tmin", "tmax"]
                instruments = ["lat", "long"]

                eff_modifiers = [s + "_lag_"+str(lags) for s in ppi_covars]
                # eff_modifiers.append("Population")
                eff_modifiers.append("log_population")
                com_causes = ["year", "month"]

                example_cols = [example_prod] + eff_modifiers + com_causes + temperatures + instruments + ["date"]
                example_df = full_df.loc[:, example_cols].reset_index(drop= True)
                example_df = example_df.rename(columns = {example_prod:"production"})

                # create log(production) column. First add 1 to all values of production to avoid taking log of 0.
                example_df.loc[:, "log_production"] = np.log(example_df.production + 1)
                example_df.loc[:, "tdiff"] = np.abs(example_df.tmax - example_df.tmin)
                

                # plot histogram of log_production
                # import matplotlib.pyplot as plt
                # plt.hist(example_df.log_production)
                # help(np.log)

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

                # treatment = "tmax" # or tavg
                instruments = ["lat", "long"]
                # instruments = ["lat", "long", "spring", "summer", "fall"]
                seasons = ["spring", "summer", "fall"]
                # seasons = ["q1", "q2", "q3"]
                # "+ M11_lag_3 * P51_lag_3 + M11_lag_3 * USD_lag_3 + P51_lag_3 * USD_lag_3 + P51_lag_3 * M11_lag_3 * USD_lag_3" + \ # interaction terms
                
                if (scale == "raw_"):
                    formula = 'production ~ 0 + year + ' + "+".join(seasons) + "+" + ' + '.join(eff_modifiers) + \
                        "+ [" + treatment + " ~ " + ' + '.join(instruments) + " ]"
                    
                else:
                    formula = 'log_production ~ 0 + year + ' + "+".join(seasons) + "+" + ' + '.join(eff_modifiers) + \
                        "+ [" + treatment + " ~ " + ' + '.join(instruments) + " ]"
                
                result_filename = "./data/user_data/01_iv_analysis/" + prov_short[i_prov] + "/" + treatment + "_" + scale +  "iv2sls_result.csv"
                dominant_naics_filename = "./data/user_data/01_iv_analysis/" + prov_short[i_prov] + "/" + treatment + "_" + scale + "dominant_naics.csv"

                iv_model = IV2SLS.from_formula(formula, data = example_df).fit(cov_type = "robust", debiased = True)
                # iv_model = IVGMMCUE.from_formula(formula, data = example_df).fit(cov_type = "robust", debiased = False)
                # print(example_prod)
                # parse(iv_model, treatment)

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