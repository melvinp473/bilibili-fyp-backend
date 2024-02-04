import base64
import io
import geopandas as gpd
import pandas as pd
import pymongo
from bson import ObjectId
from esda.getisord import G_Local
from libpysal.weights import DistanceBand
from libpysal.weights.util import min_threshold_distance, get_points_array_from_shapefile, get_points_array
import matplotlib.pyplot as plt
import numpy as np

from src.db import mongo_db_function


def spatial_analysis(shp_file_path, target_variable, data_frame, save, locations, area_level, collection,
                     mapping_variable):

    # From dataset id, find the dataset and query for unique years
    # For year in unique years:

    # Find file from github according to "year" "location" "area level"
    # URL/location/area level/year or something equivalent
    # Obtain the file path and start the spatial analysis
    # Plot the graph and store it as bytes format in an array?

    # Previously written code here
    gdf = gpd.read_file(shp_file_path)
    target_code = "code"

    matching_columns = [col for col in gdf.columns if target_code.lower() in col.lower()]
    matching = matching_columns[0]
    gdf['Value'] = np.nan

    #
    for idx, row in data_frame.iterrows():
        code = row[mapping_variable]
        value = row[target_variable]
        if pd.isna(value):
            gdf.drop(gdf[gdf[matching] == code].index, inplace=True)
        else:
            matched_row = gdf[gdf[matching] == code]

            if not matched_row.empty:
                gdf.loc[matched_row.index, 'Value'] = value
            else:
                print(f"No match found for Code: {code}")

    # for idx, row in data_frame.iterrows():
    #     code = str(row[mapping_variable])
    #     value = row[target_variable]
    #     if pd.isna(value):
    #         gdf.drop(gdf[gdf['PHA_CODE16'] == code].index, inplace=True)
    #     else:
    #         matched_row = gdf[gdf['PHA_CODE16'] == code]
    #
    #         if not matched_row.empty:
    #             gdf.loc[matched_row.index, 'Value'] = value
    #         else:
    #             print(f"No match found for Code: {code}")

    print(min_threshold_distance(get_points_array(gdf[gdf.geometry.name])))
    w = DistanceBand.from_dataframe(gdf, threshold=min_threshold_distance(get_points_array(gdf[gdf.geometry.name])))

    y = gdf['Value']
    # print(y.values)
    # print(gdf['Name'].values)
    g = G_Local(y, w, transform="R", star=True, seed=10)

    z_score = g.z_sim
    print(z_score)
    p_value = g.p_sim

    gdf['z_score'] = z_score
    gdf['p_value'] = p_value

    color_values = np.full(len(gdf), 0, dtype=object)

    hotspot_values = np.full(len(gdf), 0, dtype=object)

    # shallow blue
    condition_light_blue = np.logical_and(p_value <= 0.01, z_score < -2.58)
    color_values[condition_light_blue] = -3

    # blue
    condition_blue = np.logical_and(np.logical_and(0.01 < p_value, p_value <= 0.05),
                                    np.logical_and(-2.58 <= z_score, z_score < -1.96))
    color_values[condition_blue] = -2

    # dark blue
    condition_dark_blue = np.logical_and(np.logical_and(0.05 < p_value, p_value <= 0.1),
                                         np.logical_and(-1.96 <= z_score, z_score < -1.65))
    color_values[condition_dark_blue] = -1

    # shallow red
    condition_light_red = np.logical_and(np.logical_and(0.05 < p_value, p_value <= 0.1),
                                         np.logical_and(1.96 >= z_score, z_score > 1.65))
    color_values[condition_light_red] = 1

    # red
    condition_red = np.logical_and(np.logical_and(0.01 < p_value, p_value <= 0.05),
                                   np.logical_and(2.58 >= z_score, z_score > 1.96))
    color_values[condition_red] = 2

    # dark red
    condition_dark_red = np.logical_and(p_value <= 0.01, z_score > 2.58)
    color_values[condition_dark_red] = 3

    hotspot_values[condition_dark_red] = 1

    gdf['color'] = color_values

    gdf['is_hotspot'] = hotspot_values


    ax = gdf.plot(column='color', cmap='coolwarm', legend=True, figsize=(10, 10))
    plt.title('Getis-Ord Gi* Color-Coded by Significance')
    plt.show()
    fig = ax.get_figure()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')

    print(gdf)
    print(data_frame)

    if save:
        save_results(gdf,data_frame,collection, mapping_variable, matching)

    return_dict = {"graph": img_str}
    return return_dict


def save_results(gdf, data_frame, collection, mapping_variable, matching):

    # Frontend gives images in bytes format
    # Store it in MongoDB using the dataset id as unique identifier
    dataset_id = data_frame.iloc[0]["DATASET_ID"]
    mongo_db_function.delete_dataset(collection, dataset_id)
    for idx, row in data_frame.iterrows():
        code = row[mapping_variable]
        matched_row = gdf[gdf[matching] == code]

        if not matched_row.empty:
            hot_spot_value = matched_row.iloc[0]['is_hotspot']
            data_frame.at[idx, 'is_hotspot'] = hot_spot_value
        else:
            print(f"No match found for Code: {code}")
            data_frame.at[idx, 'is_hotspot'] = None

        document = row.to_dict()
        del document['_id']
        print(document[matching])
    data_frame.drop('_id', axis=1, inplace=True)
    dict_list = data_frame.to_dict(orient='records')
    mongo_db_function.insert_dataset(collection,dict_list)

    column_names = data_frame.columns.tolist()

    db = mongo_db_function.get_database('FIT4701')
    collection = mongo_db_function.get_collection(db, "Dataset")
    dataset_id = ObjectId(dataset_id)
    collection.update_one({'_id': dataset_id}, {"$set": {"attributes": column_names}})

"Testing code"
# file_path = '../shp/aus_pha_shape_files/pha_shape_files/2021'
# target_variable = 'Deaths from cir'
# dataset_id = '65b734a767d33b5b17c5c115'
# db = mongo_db_function.get_database('FIT4701')
# user_id = ''
# area_level = ''
# collection = mongo_db_function.get_collection(db, "Data")
#
# data = mongo_db_function.get_by_query(collection, {'DATASET_ID': dataset_id}, 'DATASET_ID')
# df = pd.DataFrame(data)
# spatial_analysis(file_path,target_variable,df,user_id,area_level)s