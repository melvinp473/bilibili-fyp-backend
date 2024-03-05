# import openpyxl
# import pymongo
# file_path = 'C:\FIT4701\FYP\AUST_MAP.xlsx'
#
# workbook = openpyxl.load_workbook(file_path)
#
# sheet_names = workbook.sheetnames
#
# first_sheet_name = sheet_names[0]
# sheet = workbook[first_sheet_name]
#
# num_rows = sheet.max_row
#
# column_a_values = []
# column_b_values = [[] for _ in range(442)]
# column_c_values = [[] for _ in range(442)]
# index = -1
# point_collect = []
# current_name = 'name'
# current_record = 'record'
#
#
# for row_number in range(2, num_rows+1):
#
#     current_name = sheet.cell(row=row_number, column=1).value
#     if current_record != current_name:
#         current_record = current_name
#         column_a_values.append(current_record)
#         index = index + 1
#         column_b_values[index].append(sheet.cell(row=row_number, column=2).value)
#         column_c_values[index].append(sheet.cell(row=row_number, column=3).value)
#
#     else:
#         column_b_values[index].append(sheet.cell(row=row_number, column=2).value)
#         column_c_values[index].append(sheet.cell(row=row_number, column=3).value)
#
# print(len(column_a_values))
#
# for i in range(len(column_a_values)):
#     location = [[] for _ in range(len(column_b_values[i]))]
#     for j in range(len(column_b_values[i])):
#         location[j]=[column_b_values[i][j],column_c_values[i][j]]
#     point_collect.append({"Name":column_a_values[i],"Location":location})
#
#
# print(len(point_collect))
#
#
# client = pymongo.MongoClient("mongodb+srv://FIT4701codebase:3v9CuZQ62QQuDO2x@cluster0.llxp5qw.mongodb.net/?retryWrites=true&w=majority")
#
#
# mydb = client["FIT4701"]
#
#
# mycollection = mydb["Coordinates"]
# result = mycollection.insert_many(point_collect)



import geopandas as gpd
#
# # 读取 Shapefile
# file_path = 'C:\FIT4701\FYP\pha_shape_files\pha_shape_files'
file_path = '../src/shp/aus_pha_shape_files/pha_shape_files/2016/PHA_2016_AUST_Gen50.shp'
gdf = gpd.read_file(file_path)
target_code = "code"

matching_columns = [col for col in gdf.columns if target_code.lower() in col.lower()]
matching = matching_columns[0]
#
import geopandas as gpd
import numpy as np
import pandas as pd

# 读取 Excel 文件
# xls_file_path = 'C:\FIT4701\FYP\pha_shape_files\(cleaned)  PHA data published 2023.xls'
# xls_data = pd.read_excel(xls_file_path)
#
# # 读取 Shapefile 文件
# shp_file_path = 'C:\FIT4701\FYP\pha_shape_files\pha_shape_files'
# gdf = gpd.read_file(shp_file_path)
# gdf['Value'] = np.nan
#
# # 将 Excel 文件中的数据与 Shapefile 文件中的数据进行匹配和操作
# for idx, row in xls_data.iterrows():
#     code = row['PHA Code']  # 假设 Excel 文件中的 Code 列是您想要匹配的列名
#     # 获取 HCC holders 列的值，如果为空则删除该行
#     value = row['HCC holders']
#     if pd.isna(value):
#         gdf.drop(gdf[gdf['Code'] == code].index, inplace=True)
#     else:
#         # 根据 Excel 文件中的 Code 查找匹配的行
#         matched_row = gdf[gdf['Code'] == code]
#         if not matched_row.empty:
#             # 进行您的操作，例如打印匹配的行数据
#             gdf.loc[matched_row.index, 'Value'] = value
#         else:
#             print(f"No match found for Code: {code}")
# print(gdf)
#
# file_path = 'C:\FIT4701\FYP\pha_shape_files\pha_shape_files'
# gdf = gpd.read_file(file_path)
#
# xls_file_path = 'C:\FIT4701\FYP\pha_shape_files\(cleaned)  PHA data published 2023.xls'
# xls_data = pd.read_excel(xls_file_path)
