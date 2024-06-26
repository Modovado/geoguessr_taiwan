import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import string
import random
from shapely.geometry import Point
from shapely import Polygon
'''
TRIAL AND ERROR
# sort by 'COUNTYID' value
# shapefile = county_shapefile.sort_values(by=['COUNTYID'])

# SELECT COLUMNS
# shapefile = county_shapefile[['COUNTYID', 'COUNTYENG', 'geometry']]
# shapefile = county_shapefile.loc[:, ['COUNTYID', 'geometry']]

# FIND geometry using COUNTYID
# print(shapefile['COUNTYID'] == 'A') # find value index
# print(shapefile.loc[shapefile['COUNTYID'] == 'A']['geometry'])
# shapefile_A = shapefile.loc[shapefile['COUNTYID'] == 'B']['geometry']


# FIND COUNTYID A-Z
# alphabetical_set = set(string.ascii_uppercase)
# county_ids = set(county_shapefile['COUNTYID'].values.tolist())
# leftout_ids = alphabetical_set - county_ids
# print(leftout_ids)
'''
from dotenv import load_dotenv, dotenv_values

# loading variables from .env file
load_dotenv()

county_filename = os.getenv("COUNTRY_SHAPE_FILE")
town_filename = os.getenv("TOWN_SHAPE_FILE")
village_filename = os.getenv("VILLAGE_SHAPE_FILE")
village_2_filename = os.getenv("VILLAGE_2_SHAPE_FILE")  # Pintung

county_shapefile = gpd.read_file(county_filename)
# town_shapefile = gpd.read_file(town_filename)
# village_shapefile = gpd.read_file(village_filename)
# village_2_shapefile = gpd.read_file(village_2_filename)

# county_shapefile
# [22 rows x 5 columns]
# ['COUNTYID', 'COUNTYCODE', 'COUNTYNAME', 'COUNTYENG', 'geometry']
# shapefile = county_shapefile
print(county_shapefile.loc[county_shapefile['COUNTYENG'] == 'Kinmen County']['geometry'].bounds)
# shapefile = county_shapefile.loc[county_shapefile['COUNTYENG'] == 'Taipei City']['geometry']

# minx, miny, maxx, maxy ->
# print(shapefile.bounds)

# town_shapefile
# [368 rows x 8 columns]
# ['TOWNID', 'TOWNCODE', 'COUNTYNAME', 'TOWNNAME', 'TOWNENG', 'COUNTYID', 'COUNTYCODE', 'geometry']
# shapefile = town_shapefile
# print((town_shapefile['TOWNENG'] == 'Kinmen County').any())
# shapefile = town_shapefile.loc[town_shapefile['TOWNENG'] == 'Green Island Township']['geometry']
# # minx, miny, maxx, maxy ->
# print(shapefile.bounds)

# village_shapefile
# [7953 rows x 11 columns]
# ['VILLCODE', 'COUNTYNAME', 'TOWNNAME', 'VILLNAME', 'VILLENG', 'COUNTYID', 'COUNTYCODE', 'TOWNID', 'TOWNCODE', 'NOTE', 'geometry']
# shapefile = village_shapefile


# shapefile = village_shapefile.loc[village_shapefile['VILLENG'] == 'Green Island Township']['geometry']
# minx, miny, maxx, maxy ->
# print(shapefile.bounds)

# village_2_shapefile
# [1 rows x 11 columns]
# shapefile = village_2_shapefile


# shapefile = shapefile.loc[:, ['TOWNENG', 'geometry']]
# print(shapefile)

# shapefile_A = shapefile.loc[shapefile['TOWNENG'] == 'Chenggong Township']['geometry'][0]
# print(shapefile_A)

# polygon = Polygon(shapefile_A)
# print(polygon)

# print(shapefile)
# shapefile.plot()
# plt.show()


# Credit: https://gis.stackexchange.com/a/207740
def generate_random(number, polygon):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points


if __name__ == '__main__':

    # n = 10
    # polygon = shapefile_A
    #
    # random_coords = generate_random(number=n, polygon=polygon)
    #
    # print(random_coords)
    print('ForsenCD')






# to csv
# shapefile.to_csv('county.csv')
# county_shapefile.to_csv('county.csv')

# print(shapefile.loc[:, ['TOWNENG', 'COUNTYID']])














