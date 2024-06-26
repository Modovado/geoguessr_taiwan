import mercantile, mapbox_vector_tile, requests, json, os
from vt2geojson.tools import vt_bytes_to_geojson


# tile_url = 'https://tiles.mapillary.com/maps/vtp/{}/2/{}/{}/{}?access_token={}'.format(tile_coverage, tile.z,
# tile.x, tile.y, access_token)
# response = requests.get(tile_url)

# url = 'https://www.google.com/'
# response = requests.get(url)  # <Response [200]>


# Presumably r.text would be preferred for textual responses, such as an HTML or XML document,
# and r.content would be preferred for "binary" filetypes, such as an image or PDF file. – dotancohen
# https://stackoverflow.com/questions/17011357/what-is-the-difference-between-content-and-text

# print(f'{response=}')
# print(f'{response.text=}')
# print(f'{response.content=}')


import numpy as np
import pandas as pd
import geopandas as gpd

county_filename = r'C:\Users\Aorus\Desktop\shapefile\county\COUNTY_MOI_1090820.shp'
county_shapefile = gpd.read_file(county_filename)

# county_shapefile
# [22 rows x 5 columns]
# ['COUNTYID', 'COUNTYCODE', 'COUNTYNAME', 'COUNTYENG', 'geometry']
# shapefile = county_shapefile

# select columns
df = county_shapefile[['COUNTYENG', 'geometry']]

# set_index
df.set_index('COUNTYENG', inplace=True)
print(df)

for index in df.index:
    county_name, polygon = index, df.loc[index]['geometry']

    print(county_name, polygon)




    # print(shapefile.iloc[index]['COUNTYENG'])

    # county_name, geometry = shapefile.iloc[index]['COUNTYENG', 'geometry']

# print(county_shapefile.loc[county_shapefile['COUNTYENG'] == 'Kinmen County']['geometry'].bounds)



