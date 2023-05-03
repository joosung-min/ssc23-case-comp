#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
from causalinference import CausalModel

style.use("fivethirtyeight")
pd.set_option("display.max_columns", 99)

# load weather_prod data
weather_prod = pd.read_csv("/home/joosungm/projects/def-lelliott/joosungm/projects/ssc23-case-comp/data/user_data/02_counterfactual_analysis/weather_prod_final.csv")
weather_prod.head()
weather_prod.shape
prov_long = weather_prod["provincename"].unique().tolist()
prov_long.sort()
prov_long

prov_short = ["AB", "BC", "MB", "NB", "NL", "NS", "ON", "PE", "QC", "SK"]

# column "is_flag" is 1 if extreme_flag is > 0, 0 otherwise
flags = ["extreme_flag", "tmax_flag", "tmin_flag"]
i_flag = int(sys.argv[1]) - 1
# i_flag = 0
flag = flags[i_flag]

weather_prod["is_flag"] = np.where(weather_prod[flag] > 0, 1, 0)
weather_prod["log_population"] = np.log(weather_prod["Population"])
weather_prod.head()
print(weather_prod["is_flag"].value_counts())
# There exists a class imbalance.

# load covariates, filter out month == 12
covar_df = pd.read_csv("/home/joosungm/projects/def-lelliott/joosungm/projects/ssc23-case-comp/data/user_data/_covariates/ppi_and_usd_imputed.csv")

# - group by year, compute the average of each covariate.
covar_df = covar_df.drop(columns = ["date", "month"]).reset_index(drop = True)
covar_df = covar_df.groupby("year").mean().reset_index()
covar_df.head()

# - min max scale each column of covar_df
from sklearn.preprocessing import MinMaxScaler
covar_sub = covar_df.drop(columns = ["year"])
# covar_sub = covar_df.copy()
covar_scaled = pd.DataFrame(MinMaxScaler().fit_transform(covar_sub), columns = covar_sub.columns)

# - Perform PCA on the scaled covariates
from sklearn.decomposition import PCA
n_pca = 5
pca = PCA(n_components=n_pca)
pca.fit(covar_scaled)
print(np.sum(pca.explained_variance_ratio_)) # 5PCs explain over 98% of the variance in the data
covar_pca = pd.DataFrame(pca.transform(covar_scaled), columns = ["cPC" + str(i) for i in range(1, n_pca+1)])

# - Add year column back
covar_pca = pd.concat([covar_df.loc[:, "year"], covar_pca], axis = 1)
covar_pca.head()

# - merge covar_pca to weather_prod by year
weather_prod_df = pd.merge(weather_prod, covar_pca, on = "year", how = "left")
weather_prod_df.head()


# - remove "production_in_division_" from column names
weather_prod_df.columns = [col.replace("production_in_division_", "") for col in weather_prod_df.columns]

# - extract columns that start with "X"
prod_cols = [col for col in weather_prod.columns if col.startswith("X")]

covar_cols = ["year", "log_population", "lat", "lon", "cPC1", "cPC2", "cPC3", "cPC4", "cPC5"]

for p, prov_s in enumerate(prov_short):
    # p = 0
    # prov_s = prov_short[p]
    prov_l = prov_long[p]

    weather_prod_sub = weather_prod_df.loc[weather_prod_df["provincename"] == prov_l, :]
    # weather_prod_sub.head()
    prod_ate = pd.DataFrame(columns = ["Industry", "mean(ATE)", "se(ATE)", "lower_95ci", "upper_95ci", "sig"])

    for i, prod in enumerate(prod_cols):
        
        # prod = prod_cols[14]
        # print(prod)

        T = "is_flag"
        # Y = prod_cols[9] # X11.Agriculture.forestry.fishing.hunting.21.Mining.quarrying.and.oil.and.gas.extraction
        X = covar_cols

        df_sub = weather_prod_sub.loc[:, [T, prod] + X].dropna()
        df_sub["log_prod"] = np.log(df_sub[prod] + 1)
        Y = "log_prod"

        # - Use bootstrapping to compute the mean and standard error of ATE
        from joblib import Parallel, delayed # for parallel processing

        # define function that computes the IPTW estimator
        def run_ps(df, X, T, y):
            
            # estimate the propensity score using logistic regression.
            # - make prediction to get propensity score.
            ps = LogisticRegression(C=1e6, max_iter = 1000).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
            
            weight = (df[T]-ps) / (ps*(1-ps)) # define the weights
            return np.mean(weight * df[y]) # compute the ATE

        np.random.seed(42)

        # run 1000 bootstrap samples
        bootstrap_sample = 1000
        ates = Parallel(n_jobs = 10)(delayed(run_ps)(df_sub.sample(frac=1, replace=True), X, T, Y)
                                for _ in range(bootstrap_sample))
        ates = np.array(ates)


        # - The ATE is then the mean of the bootstrap samples, and CI is the 2.5% and 97.5% quantiles of the bootstrap samples.
        mean_ate = np.round(np.mean(ates), 3) # IPTW (inverse probability of treatment weighted) estimator
        se_ate = np.round(np.std(ates), 3)
        lowerci_ate = np.round(np.percentile(ates, 2.5), 3)
        upperci_ate = np.round(np.percentile(ates, 97.5), 3)
        
        is_sig = (lowerci_ate > 0) | (upperci_ate < 0) # is_sig is True if 0 is not in the 95% CI
        
        # - Store result
        prod_ate.loc[i] = [prod, mean_ate, se_ate, lowerci_ate, upperci_ate, is_sig]
        # prod_ate

    result_filename = "/home/joosungm/projects/def-lelliott/joosungm/projects/ssc23-case-comp/data/user_data/02_counterfactual_analysis/" + prov_s + "/" + "result_" + flag + ".csv"
    prod_ate.to_csv(result_filename, index = False)
