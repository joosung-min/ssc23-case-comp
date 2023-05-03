import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from meteostat import Stations, Daily, Point, Monthly
import geopandas as gpd
import os
import multiprocessing as mp

os.chdir("/home/joosungm/projects/def-lelliott/joosungm/projects/ssc23-case-comp")

prov_long = ["Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland and Labrador", "Nova Scotia", "Ontario", "Prince Edward Island", "Quebec", "Saskatchewan"]

prov_short = ["AB", "BC", "MB", "NB", "NL", "NS", "ON", "PE", "QC", "SK"]

for i_prov in range(len(prov_short)):
# for i_prov in range(1):
# def get_temp(i_prov):
    # i_prov = 1
    prov_long = ["Alberta", "British Columbia", "Manitoba", "New Brunswick", "Newfoundland and Labrador", "Nova Scotia", "Ontario", "Prince Edward Island", "Quebec", "Saskatchewan"]

    prov_short = ["AB", "BC", "MB", "NB", "NL", "NS", "ON", "PE", "QC", "SK"]
    print(prov_long[i_prov])

    production_filename = "./data/Productivity per NAICS within region/4.c_production_in_CSD_in" + prov_long[i_prov] + ".csv"
    production = pd.read_csv(production_filename)

    # Obtain centroid for each CSD
    geo_filename = "./data/geojson_files/1.a_census_data_" + prov_short[i_prov] + "_CSD_geometry_only.geojson"
    geo_df = gpd.read_file(geo_filename)
    geo_df["long"] = geo_df["geometry"].centroid.x
    geo_df["lat"] = geo_df["geometry"].centroid.y
    
    # extract the maximum, minimum latitude and longitude from the geo_df geometry
    geo_df["max_lat"] = geo_df["geometry"].bounds["maxy"]
    geo_df["min_lat"] = geo_df["geometry"].bounds["miny"]
    geo_df["max_long"] = geo_df["geometry"].bounds["maxx"]
    geo_df["min_long"] = geo_df["geometry"].bounds["minx"]
    
    geo_df = geo_df.astype({"GeoUID": "int64"})
    # geo_df.head(2)
    
    prod = production.merge(geo_df.loc[:, ["GeoUID", "lat", "long", "max_lat", "min_lat", "max_long", "min_long"]], on = "GeoUID", how = "left")
    prod["month"] = pd.to_datetime(prod["Date"]).dt.month
    prod["year"] = pd.to_datetime(prod["Date"]).dt.year
    # drop rows that have lat == NaN
    prod = prod.dropna(subset = ["lat", "long"])
    # prod.shape
    # print(np.min(prod["year"]))
    # print(np.max(prod["year"]))

    GeoUIDs = prod["GeoUID"].unique()
    start = datetime(1997, 1, 1)
    end = datetime(2021, 12, 31)
    data_all = pd.DataFrame(columns = ["year", "month", "tavg", "tmin", "tmax", "GeoUID", "lat", "long"])

    temp_missing_ids = []

    for i, id in enumerate(GeoUIDs):
        # print(id)
        # Download daily temperature data for each GeoUID
        
        # i = 1
        # id = GeoUIDs[i]
        # id = 5949802
        # - set location
        error = False
        geoUID_lat = prod.loc[prod["GeoUID"] == id, "lat"].values[0]
        geoUID_long = prod.loc[prod["GeoUID"] == id, "long"].values[0]

        location = Point(geoUID_lat, geoUID_long)
        
        # - fetch data
        data = Daily(location, start, end)
        data = data.fetch().loc[:, ["tavg", "tmin", "tmax"]]
        data.reset_index(inplace = True)
        # data.shape
        
        # data.head(5)    
        if data.shape[0] == 0:

            lat_seq = np.linspace(prod.loc[prod["GeoUID"] == id, "min_lat"].values[0], prod.loc[prod["GeoUID"] == id, "max_lat"].values[0], 100)
            long_seq = np.linspace(prod.loc[prod["GeoUID"] == id, "min_long"].values[0], prod.loc[prod["GeoUID"] == id, "max_long"].values[0], 100)
            
            # create a grid of latitudes and longitudes
            lat_grid, long_grid = np.meshgrid(lat_seq, long_seq)
            lat_grid = lat_grid.flatten()
            long_grid = long_grid.flatten()

            # create a dataframe of latitudes and longitudes
            lat_long_df = pd.DataFrame({"lat": lat_grid, "long": long_grid})

            # shuffle rows
            lat_long_df = lat_long_df.sample(frac = 1).reset_index(drop = True)

            # from shapely.geometry import Point as geoPoint
            # lat_long_df["Point"] = lat_long_df.apply(lambda x: geoPoint(x["lat"], x["long"]), axis = 1)
            # lat_long_point = gpd.GeoDataFrame(lat_long_df.loc[:,"Point"], geometry = "Point", crs=4326)            
            # geo_poly = gpd.GeoDataFrame(geo_df.loc[geo_df["GeoUID"] == id, "geometry"], geometry="geometry")
            
            # from geopandas.tools import sjoin
            # pointInMultiPoly = sjoin(lat_long_point, geo_poly, how = "left")
            # pointInMultiPoly
            # grouped = pointInMultiPoly.groupby("index_right")
            # list(grouped)
            
            for j in range(lat_long_df.shape[0]):
                # j = 10
                geoUID_lat = lat_long_df.loc[j, "lat"]
                geoUID_long = lat_long_df.loc[j, "long"]
                
                location = Point(geoUID_lat, geoUID_long)
                
                data = Daily(location, start, end)
                data = data.fetch().loc[:, ["tavg", "tmin", "tmax"]]
                data.reset_index(inplace = True)
                
                if data.shape[0] > 0:
                    print("found temperature for {}".format(id))
                    break

                else:
                    print("no temperature for {}".format(id))
                    # save id to a list
                    temp_missing_ids.append(id)
                    error = True
                    break
        if (error):
            continue
        # if tavg is NaN, then replace it with the average of tmin and tmax
        data.loc[data["tavg"].isna(), "tavg"] = (data.loc[data["tavg"].isna(), "tmin"] + data.loc[data["tavg"].isna(), "tmax"])/2
        # data.head(5)
        try:
            data.loc[:, "month"] = pd.to_datetime(data.loc[:, "time"]).dt.month
            data.loc[:, "year"] = pd.to_datetime(data.loc[:, "time"]).dt.year

        except:
            print("time error at {}".format(id))
            continue
        
        # - take tavg from data_mean, tmin from data_min, tmax from data_max and merge them as a data frame named data2
        # print(prov_long[i_prov])
        data_mean = data.groupby(["year", "month"]).mean(numeric_only = True).reset_index()
        data_max = data.groupby(["year", "month"]).max(numeric_only = True).reset_index()
        data_min = data.groupby(["year", "month"]).min(numeric_only = True).reset_index()
        try:
            data2 = data_mean.loc[:, ["year", "month", "tavg"]].copy()
        except KeyError:
            print("KeyError at {id} in {prov}".format(id = id, prov = prov_long[i_prov]))
            data["tavg"] = (data["tmin"] + data["tmax"])/2
            data_mean = data.groupby(["year", "month"]).mean(numeric_only = True).reset_index()
            data_max = data.groupby(["year", "month"]).max(numeric_only = True).reset_index()
            data_min = data.groupby(["year", "month"]).min(numeric_only = True).reset_index()
            data2 = data_mean.loc[:, ["year", "month", "tavg"]].copy()

        data2.loc[:, "tmin"] = data_min["tmin"]
        data2.loc[:, "tmax"] = data_max["tmax"]
        data2.loc[:, "GeoUID"] = id
        data2.loc[:, "lat"] = geoUID_lat
        data2.loc[:, "long"] = geoUID_long
        
        data_all = pd.concat([data_all, data2], axis = 0)
    
    
    # merge the temperature data to the production data
    # prod_temp = prod.merge(data_all, on = ["GeoUID", "year", "month"], how = "left")
    # prod_temp.to_csv(dir_name + "/prod_temp.csv", index = False)

    for missing_id in temp_missing_ids:
        # missing_id = temp_missing_ids[1]
        
        # find the GeoUID that has the closest latitude and longitude to the missing id, other than itself
        missing_id_lat = prod.loc[prod["GeoUID"] == missing_id, "lat"].values[0]
        missing_id_long = prod.loc[prod["GeoUID"] == missing_id, "long"].values[0]
        
        data_all2 = data_all.copy()
        dist = (data_all2["lat"] - missing_id_lat)**2 + (data_all2["long"] - missing_id_long)**2
        dist = dist.astype(float)
        
        data_all2["dist"] = np.sqrt(dist)
        
        closest_id = data_all2.loc[data_all2["dist"] == np.min(data_all2["dist"]), "GeoUID"].values[0]
        print("closest id to {} is {}".format(missing_id, closest_id))

        # find the closest id's temperature data
        closest_id_temp = data_all.loc[data_all.loc[:, "GeoUID"] == closest_id, :]
        closest_id_temp.loc[:, "GeoUID"] = missing_id
        data_all = pd.concat([data_all, closest_id_temp], axis = 0)

    # merge the temperature data and save the final output
    # - exclude columns lat, long from prod first.
    prov = prov_short[i_prov]
    dir_name = "./data/user_data/01_iv_analysis/" + prov
    
    prod = prod.loc[:, ~prod.columns.isin(["lat", "long"])]
    prod_temp = prod.merge(data_all, on = ["GeoUID", "year", "month"], how = "left")
    prod_temp.to_csv(dir_name + "/prod_temp.csv", index = False)

print("done!")
