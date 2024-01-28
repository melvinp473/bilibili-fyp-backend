import base64
import io
import geopandas as gpd
import pandas as pd
from esda.getisord import G_Local
from libpysal.weights import DistanceBand
from libpysal.weights.util import min_threshold_distance, get_points_array_from_shapefile, get_points_array
import matplotlib.pyplot as plt
import numpy as np


def spatial_analysis(file, target_variable, dataset_id, locations, area_level):

    # From dataset id, find the dataset and query for unique years
    # For year in unique years:

        # Find file from github according to "year" "location" "area level"
        # URL/location/area level/year or something equivalent
        # Obtain the file path and start the spatial analysis
        # Plot the graph and store it as bytes format in an array?

        # Previously written code here
        # file_path = 'C:\\Users\\Asus\\Downloads\\pha_aust_2021\\pha_map\\PHA_2021_Aust_GDA2020_Gen50.shp'
        # gdf = gpd.read_file(file_path)
        #
        # xls_file_path = 'C:\\Users\\Asus\\Documents\\FIT4701\\FYP extension\\Aus Data\\(cleaned)  PHA data published 2023.xls'
        # xls_data = pd.read_excel(xls_file_path)
        #
        # gdf['Value'] = np.nan
        #
        # for idx, row in xls_data.iterrows():
        #     code = row['PHA Code']
        #     value = row['Deaths from circulatory system diseases 0 to 74 years']
        #     if pd.isna(value):
        #         gdf.drop(gdf[gdf['Code'] == code].index, inplace=True)
        #     else:
        #         matched_row = gdf[gdf['Code'] == code]
        #
        #         if not matched_row.empty:
        #             gdf.loc[matched_row.index, 'Value'] = value
        #         else:
        #             print(f"No match found for Code: {code}")
        #
        # w = DistanceBand.from_dataframe(gdf, threshold=min_threshold_distance(get_points_array(gdf[gdf.geometry.name])))
        #
        # y = gdf['Value']
        # g = G_Local(y, w, transform="R", star=True, seed=10)
        #
        # z_score = g.z_sim
        # p_value = g.p_sim
        #
        # gdf['z_score'] = z_score
        # gdf['p_value'] = p_value
        #
        # # 根据条件设置颜色
        # color_values = np.full(len(gdf), 0, dtype=object)
        #
        # # 浅蓝
        # condition_light_blue = np.logical_and(p_value <= 0.01, z_score < -2.58)
        # color_values[condition_light_blue] = -3
        #
        # # 蓝
        # condition_blue = np.logical_and(np.logical_and(0.01 < p_value, p_value <= 0.05),
        #                                 np.logical_and(-2.58 <= z_score, z_score < -1.96))
        # color_values[condition_blue] = -2
        #
        # # 深蓝
        # condition_dark_blue = np.logical_and(np.logical_and(0.05 < p_value, p_value <= 0.1),
        #                                      np.logical_and(-1.96 <= z_score, z_score < -1.65))
        # color_values[condition_dark_blue] = -1
        #
        # # 浅红
        # condition_light_red = np.logical_and(np.logical_and(0.05 < p_value, p_value <= 0.1),
        #                                      np.logical_and(1.96 >= z_score, z_score > 1.65))
        # color_values[condition_light_red] = 1
        #
        # # 红
        # condition_red = np.logical_and(np.logical_and(0.01 < p_value, p_value <= 0.05),
        #                                np.logical_and(2.58 >= z_score, z_score > 1.96))
        # color_values[condition_red] = 2
        #
        # # 深红
        # condition_dark_red = np.logical_and(p_value <= 0.01, z_score > 2.58)
        # color_values[condition_dark_red] = 3
        #
        # gdf['color'] = color_values
        #
        # ax = gdf.plot(column='color', cmap='coolwarm', legend=True, figsize=(10, 10))
        # plt.title('Getis-Ord Gi* Color-Coded by Significance')
        # plt.show()
        # fig = ax.get_figure()
        # buffer = io.BytesIO()
        # fig.savefig(buffer, format='png')
        # buffer.seek(0)
        # img_str = base64.b64encode(buffer.read()).decode('utf-8')
        # print(img_str)
        #
        # for idx, row in gdf.iterrows():
        #     area_name = row['Code']
        #     value = row['Value']
        #     z_score = g.z_sim[idx]
        #     p_value = g.p_sim[idx]
        #     color = row['color']
        #
        #     print(f'{area_name}: Value={value}, Z-Score={z_score}, P-Value={p_value}，Color={color}')

    # After storing graph for each year, return the array to the frontend

    return None


def save_results(images, dataset_id):

    # Frontend gives images in bytes format
    # Store it in MongoDB using the dataset id as unique identifier

    return None