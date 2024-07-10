"""
Download Mapillary images a JPGs
Credits: cbeddow [https://gist.github.com/cbeddow/79d68aa6ed0f028d8dbfdad2a4142cf5]
"""
import mercantile
import requests
import json
import os
from vt2geojson.tools import vt_bytes_to_geojson
from shapely.geometry import Point, Polygon
import geopandas as gpd
from dotenv import load_dotenv, dotenv_values
from tqdm import tqdm, trange
from addict import Dict
# loading variables from .env file
load_dotenv()

# Mapillary
# define an empty geojson as output
# output = {"type": "FeatureCollection", "features": []}
# vector tile endpoints -- change this in the API request to reference the correct endpoint
tile_coverage: str = 'mly1_public'
# tile layer depends on which vector tile endpoints:
# 1. if map features or traffic signs, it will be "point" always
# 2. if looking for coverage, it will be "image" for points, "sequence" for lines, or "overview" for far zoom
tile_layer: str = 'image'
# Mapillary access token -- user should provide their own
access_token = os.getenv("MAPILLARY_STREET_VIEW_API_KEY")

# county_shapefile
county_filename = os.getenv("COUNTRY_SHAPE_FILE")
county_shapefile = gpd.read_file(county_filename)

# ['COUNTYID', 'COUNTYCODE', 'COUNTYNAME', 'COUNTYENG', 'geometry']
# select columns
df = county_shapefile[['COUNTYENG', 'geometry']]

image_amount: int = 40_000  # per county

log = Dict() # log

# Lienchiang County all
#      Yilan County tiles [1049 / -----]
#   Changhua County tiles [ 186 /  ----]
#     Nantou County tiles [  96 /  ----]
#     Yunlin County tiles [  467/  ----]
#     Keelung  City tiles [  175/  ----]
#      Taipei  City tiles [   33/  ----]
#   New Taipei City tiles [   136/  ----] #24xxx


# 　https://stackoverflow.com/a/65873361
#　EXCLUDING
counties_exclude:list[str] = ['Lienchiang County',
                              'Yilan County',
                              'Changhua County',
                              'Nantou County',
                              'Yunlin County',
                              'Keelung City',
                              'Taipei City',
                              'Tainan City',
                              'Taoyuan City',
                              # 'Miaoli County' currently
                              ]
# append county_exclude from log
with open('log.json') as f:
    counties_log = json.load(f)

for county in counties_log.keys():
    if county not in counties_exclude:
        counties_exclude.append(county)

df = df[~(df['COUNTYENG'].isin(counties_exclude))]
df = df.reset_index(drop=True)
np_df = df.to_numpy()

#　 INCLUDING(one only)
# county_include:str = 'Yilan County'
# df = df[df["COUNTYENG"] == county_include]
# county_select:str  = ''
# tile_exclude: int = 0
# feature_exclude: int = 0


if __name__ == '__main__':
    for county_index, (county_name, polygon) in enumerate(
            tqdm(np_df, total=len(np_df), leave=False, position=0, desc='Counties')):
        tqdm.write(f'{county_name=}')

        west, south, east, north = polygon.bounds  # minx, miny, maxx, maxy
        tiles = list(mercantile.tiles(west, south, east, north, 14))

        image_progress_bar = tqdm(total=image_amount, leave=False, position=3, desc='Downloaded Images')

        for tile_index, tile in enumerate(tqdm(tiles, total=len(tiles), leave=False, position=1, desc='Tiles')):

            if image_progress_bar.n >= image_amount:
                break

            # log tile_index
            log[f'{county_name}'].tile_index = tile_index


            tile_url = ('https://tiles.mapillary.com/maps/vtp/{}/2/{}/{}/{}?access_token={}'
                        .format(tile_coverage,
                                tile.z,
                                tile.x,
                                tile.y,
                                access_token))

            response = requests.get(tile_url)
            # turn request.content which is "binary" filetype into geojson form
            data = vt_bytes_to_geojson(response.content, tile.x, tile.y, tile.z, layer=tile_layer)

            for feature_index, feature in enumerate(
                    tqdm(data['features'], total=len(data['features']), leave=False, position=2, desc='Feature')):

                if image_progress_bar.n >= image_amount:
                    break

                else:
                    # get lng,lat of each feature
                    lng = feature['geometry']['coordinates'][0]
                    lat = feature['geometry']['coordinates'][1]
                    pnt = Point(lng, lat)

                    if polygon.contains(pnt):

                        # create a folder for each unique sequence ID to group images by sequence
                        sequence_id = feature['properties']['sequence_id']
                        if not os.path.exists(f'dataset/{county_name}/{sequence_id}'):
                            os.makedirs(f'dataset/{county_name}/{sequence_id}')

                        # request the URL of each image
                        image_id = feature['properties']['id']
                        header = {'Authorization': 'OAuth {}'.format(access_token)}
                        fields = ('altitude, '
                                  'atomic_scale, '
                                  'camera_parameters, '
                                  'camera_type, '
                                  'captured_at, '
                                  'compass_angle, '
                                  'computed_altitude, '
                                  'computed_compass_angle, '
                                  'computed_geometry, '
                                  'computed_rotation, '
                                  'creator, '
                                  'exif_orientation, '
                                  'geometry, '
                                  'height, '
                                  'is_pano, '
                                  'make, '
                                  'model, '
                                  'thumb_2048_url, '
                                  'thumb_1024_url, '
                                  'merge_cc, '
                                  'mesh, '
                                  'sequence, '
                                  'sfm_cluster, '
                                  'width, '
                                  'detections')

                        url = 'https://graph.mapillary.com/{}?fields={}'.format(image_id, fields)
                        r = requests.get(url, headers=header)
                        data = r.json()

                        if 'thumb_2048_url' in data:
                            image_url = data['thumb_2048_url']
                        elif 'thumb_1024_url' in data:
                            image_url = data['thumb_1024_url']
                            tqdm.write(f'{county_name} / {sequence_id} / {image_id}.jpg is 1024')
                        else:  # there's 256 left, but we do not want any image smaller than 384
                            image_url = None

                        if image_url:
                            # save each image with ID as filename to directory by sequence ID
                            with open(f'dataset/{county_name}/{sequence_id}/{image_id}.jpg', 'wb') as handler:
                                image_data = requests.get(image_url, stream=True).content
                                handler.write(image_data)

                            # save a local geojson with the filtered data
                            with open(f'dataset/{county_name}/{sequence_id}/{image_id}.geojson', 'w') as f:
                                json.dump(data, f)

                            image_progress_bar.update(1)

                            # log feature
                            log[f'{county_name}'].feature_index = feature_index

            if image_progress_bar.n >= image_amount:
                with open('log.json', 'w') as f:
                    json.dump(log, f)
                break

