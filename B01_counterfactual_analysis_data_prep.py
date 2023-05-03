#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

pd.set_option("print.max_columns", 99)

# Here, we create a dataset that includes:
# 
# * lat, long of the weather stations.
# * year
# * tmin
# * tmax
# * tavg
# * extMax
# * extMin
# * prcp
# * is_flag: whether there was an extreme weather event in that year.
# 

# Load data
weather = pd.read_csv("/home/joosungm/projects/def-lelliott/joosungm/projects/ssc23-case-comp/data/climate_data/weather_Station_data.csv")
print(weather.columns)

# Extract columns
weather_columns = ["Station Name", "Longitude (x)", "Latitude (y)", "Year", "Month", 
"Mean Max Temp (°C)", 'Mean Min Temp (°C)', 'Mean Temp (°C)', 
'Extr Max Temp (°C)', 'Extr Max Temp Flag', 'Extr Min Temp (°C)', 'Extr Min Temp Flag',
'Total Precip (mm)', 'Total Precip Flag'
]

weather = weather.loc[:, weather_columns]
print(weather.head())


weather2 = weather.rename(columns = {
    "Longitude (x)":"lon", "Latitude (y)":"lat", "Year":"year", "Month":"month",
    "Mean Max Temp (°C)":"tmax", "Mean Min Temp (°C)":"tmin", "Mean Temp (°C)":"tavg", 
    "Extr Max Temp (°C)":"tmax_ext", "Extr Min Temp (°C)":"tmin_ext",
    "Extr Max Temp Flag":"tmax_ext_flag", "Extr Min Temp Flag":"tmin_ext_flag",
    "Total Precip (mm)":"precip", "Total Precip Flag":"precip_flag"})
weather2.head()


print(weather2.tmax_ext_flag.unique())
print(weather2.tmin_ext_flag.unique()) 
print(weather2.precip_flag.unique())
# I: incomplete
# S: more than 1 occurrence
# E: estimated
# B: more than 1 occurence & estimated
# M: missing
# T: trace; value is zero.


# extreme_flag == 1 if one of tmax_ext_flag, tmin_ext_flag is either one of S, E, and B
weather2["extreme_flag"] = np.where((weather2.tmax_ext_flag.isin(["S", "E", "B"])) | (weather2.tmin_ext_flag.isin(["S", "E", "B"])), 1, 0)

# tmax_flag == 1 if tmax_ext_flag is either one of S, E, and B
weather2["tmax_flag"] = np.where(weather2.tmax_ext_flag.isin(["S", "E", "B"]), 1, 0)

# tmin_flag == 1 if tmin_ext_flag is either one of S, E, and B
weather2["tmin_flag"] = np.where(weather2.tmin_ext_flag.isin(["S", "E", "B"]), 1, 0)

# total_flag = sum of tmax_flag, tmin_flag, and precip_flag
weather2["total_flag"] = weather2.tmax_flag + weather2.tmin_flag

# group by Station Name, year, lat, long, and sum extreme_flag.
weather3 = weather2.groupby(["Station Name", "year", "lat", "lon"], as_index=False).agg({"extreme_flag":"sum", "tmax_flag":"sum", "tmin_flag":"sum", "total_flag":"sum"})
weather3.head()


province_prod = pd.DataFrame()
provinces = ["AB", "BC", "MB", "NB", "NL", "NS", "ON", "PE", "QC", "PE", "SK"]
for pr in provinces:
    # pr = "AB"
    prod_temp_filename = "/home/joosungm/projects/def-lelliott/joosungm/projects/ssc23-case-comp/data/user_data/01_iv_analysis/" + pr + "/prod_temp.csv"

    prod_temp = pd.read_csv(prod_temp_filename)
    # Exclude "production_in_division_" from all column names
    prod_temp.columns = [col.replace("production_in_division_", "") for col in prod_temp.columns]
    prod_temp.rename(columns = {"lat":"prod_lat", "long":"prod_lon"}, inplace = True)
    # prod_temp.head()
    
    # extract GeoUID, year, month, and all columns that starts with "X" 
    prod_temp_sub = prod_temp.loc[:, ["provincename", "GeoUID", "year"] + [col for col in prod_temp.columns if col.startswith("X")]]
    
    # groupby GeoUID and year, sum over all columns that starts with "X"
    prod_temp_sub = prod_temp_sub.groupby(["provincename", "GeoUID", "year"], as_index = False).sum().reset_index(drop = True)
    # prod_temp_sub.head()

    # re-attach Dominant_NAICS
    prod_temp_latest = prod_temp.loc[prod_temp.month == 12, ["provincename", "GeoUID", "year", "Dominant_NAICS", "prod_lat", "prod_lon", "Population"]].reset_index(drop = True)
    prod_temp_sub = prod_temp_sub.merge(prod_temp_latest, on = ["provincename", "GeoUID", "year"], how = "left")
    
    # concat prod_temp to province_prod by row
    province_prod = pd.concat([province_prod, prod_temp_sub], axis = 0)

print(province_prod.shape)

# drop rows that have NaN Dominant_NAICS
province_prod = province_prod.dropna(subset = ["Dominant_NAICS"]).reset_index(drop = True)
province_prod.tail()

# save province_prod
province_prod.to_csv("/home/joosungm/projects/def-lelliott/joosungm/projects/ssc23-case-comp/data/user_data/02_counterfactual_analysis/province_prod.csv", index = False)

# filter out unique set of lat, lon
weather_unique = weather3.drop_duplicates(subset = ["lat", "lon"]).reset_index(drop = True)[["Station Name", "lat", "lon"]]
print(weather_unique.head())
print(weather_unique.shape)

# filter out unique GeoUID and their prod_lat, prod_lon from province_prod, drop NaN
# province_prod = pd.read_csv("/home/joosungm/projects/def-lelliott/joosungm/projects/ssc23-case-comp/data/user_data/02_counterfactual_analysis/province_prod.csv")
province_prod_unique = province_prod.drop_duplicates(subset = ["GeoUID", "prod_lat", "prod_lon"]).dropna().reset_index(drop = True)[["provincename", "GeoUID", "prod_lat", "prod_lon"]]
print(province_prod_unique.head())
print(province_prod_unique.shape)


# Measure ucleadian distance between each weather station and each GeoUID
def distance(lat1, lon1, lat2, lon2):

    km = np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

    return km

# Measure distance between two lat/lon pairs
# def distance(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    # lat1 = np.radians(lat1)
    # lon1 = np.radians(lon1)
    # lat2 = np.radians(lat2)
    # lon2 = np.radians(lon2)
    # # Find the differences
    # dlat = lat2 - lat1
    # dlon = lon2 - lon1
    # # Apply the formula
    # a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    # c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) # great circle distance in radians
    # # Convert to kilometers
    # km = 6371 * c
    # return km


# For each GeoUID, find the closest station name and its lat, lon.
# - Add weather Station name, lat, lon, distance columns to a copy of province_prod_unique.

province_prod_unique2 = province_prod_unique.copy()
province_prod_unique2["weather_station_name"] = ""
province_prod_unique2["weather_station_lat"] = 0
province_prod_unique2["weather_station_lon"] = 0
province_prod_unique2["weather_station_distance"] = 0

for i in range(province_prod_unique2.shape[0]):
    # get GeoUID, prod_lat, prod_lon
    # i = 1
    geo_uid = province_prod_unique2.loc[i, "GeoUID"]
    prod_lat = province_prod_unique2.loc[i, "prod_lat"]
    prod_lon = province_prod_unique2.loc[i, "prod_lon"]
    
    # get weather station name, lat, lon
    weather_station_name = weather_unique.loc[weather_unique.index[0], "Station Name"]
    weather_station_lat = weather_unique.loc[weather_unique.index[0], "lat"]
    weather_station_lon = weather_unique.loc[weather_unique.index[0], "lon"]
    # get distance
    weather_station_distance = distance(prod_lat, prod_lon, weather_station_lat, weather_station_lon)
    # loop through weather_unique to find the closest station name, lat, lon
    for j in range(weather_unique.shape[0]):
        # j = 1
        # get weather station name, lat, lon
        weather_station_name_temp = weather_unique.loc[weather_unique.index[j], "Station Name"]
        weather_station_lat_temp = weather_unique.loc[weather_unique.index[j], "lat"]
        weather_station_lon_temp = weather_unique.loc[weather_unique.index[j], "lon"]
        # get distance
        weather_station_distance_temp = distance(prod_lat, prod_lon, weather_station_lat_temp, weather_station_lon_temp)
        # if distance is smaller, update weather_station_name, lat, lon, distance
        if weather_station_distance_temp < weather_station_distance:
            weather_station_name = weather_station_name_temp
            weather_station_lat = weather_station_lat_temp
            weather_station_lon = weather_station_lon_temp
            weather_station_distance = weather_station_distance_temp
    # update province_prod_unique2
    province_prod_unique2.loc[i, "weather_station_name"] = weather_station_name
    province_prod_unique2.loc[i, "weather_station_lat"] = weather_station_lat
    province_prod_unique2.loc[i, "weather_station_lon"] = weather_station_lon
    province_prod_unique2.loc[i, "weather_station_distance"] = weather_station_distance


province_prod_unique2 = province_prod_unique2.dropna().reset_index(drop = True)
province_prod_unique2.head()
province_prod_unique2.shape

# - Extract weather_station_name and provincename
station_province = province_prod_unique2[["weather_station_name", "provincename"]].drop_duplicates().reset_index(drop = True)
station_province.shape

# - for each weather_station_name, compute the number of provincename
station_province["num_provincename"] = 0
for i in range(station_province.shape[0]):
    # i = 0
    weather_station_name = station_province.loc[i, "weather_station_name"]
    # weather_station_name
    num_provincename = station_province[station_province["weather_station_name"] == weather_station_name].shape[0]
    station_province.loc[i, "num_provincename"] = num_provincename

dup_provinces = station_province[station_province["num_provincename"] > 1]["weather_station_name"].tolist()

for station in dup_provinces:
    # station = dup_provinces[0]
    temp_dup = province_prod_unique2[province_prod_unique2["weather_station_name"] == station].groupby(["provincename"]).count().reset_index()
    dominant_prov = temp_dup[temp_dup["GeoUID"] == temp_dup["GeoUID"].max()]["provincename"].tolist()[0]

    # drop all the rows with weather_station_name == station and provincename != dominant_prov
    province_prod_unique2 = province_prod_unique2[~((province_prod_unique2["weather_station_name"] == station) & (province_prod_unique2["provincename"] != dominant_prov))].reset_index(drop = True)


# - left join province_prod_unique2 to province_prod by GeoUID
province_prod2 = pd.merge(province_prod, province_prod_unique2[["GeoUID", "weather_station_name"]], on = "GeoUID", how = "left")
province_prod2_cols1 = ["weather_station_name", "year", "provincename"]
province_prod2_cols2 = [col for col in prod_temp.columns if col.startswith("X")] + ["Population"] # productions and Population
province_prod2_cols = province_prod2_cols1 + province_prod2_cols2
province_prod2_cols
province_prod2.head()

# - group by weather_station_name and year, sum up all the columns in province_prod2_cols
province_prod3 = province_prod2[province_prod2_cols].dropna().groupby(["weather_station_name", "year", "provincename"]).sum().reset_index()
province_prod3.head()
province_prod3.shape

# merge province_prod4 to weather3 by weather_station_name and year
weather_prod_final = pd.merge(weather3, province_prod3, left_on = ["Station Name", "year"], right_on = ["weather_station_name", "year"], how = "left").dropna().reset_index()
weather_prod_final.shape
weather_prod_final.provincename.unique()

weather_prod_final.to_csv("/home/joosungm/projects/def-lelliott/joosungm/projects/ssc23-case-comp/data/user_data/02_counterfactual_analysis/weather_prod_final.csv", index = False)

weather_prod_final.head()