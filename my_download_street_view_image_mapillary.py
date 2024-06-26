"""
Download Mapillary images a JPGs

Credits: cbeddow
https://gist.github.com/cbeddow/79d68aa6ed0f028d8dbfdad2a4142cf5
"""

from shapely.geometry import Point, Polygon
import mercantile, mapbox_vector_tile, requests, json, os
from vt2geojson.tools import vt_bytes_to_geojson
from dotenv import load_dotenv, dotenv_values

# loading variables from .env file
load_dotenv()

# define an empty geojson as output
output = {"type": "FeatureCollection", "features": []}
# output = {}

# vector tile endpoints -- change this in the API request to reference the correct endpoint
tile_coverage = 'mly1_public'

# tile layer depends on which vector tile endpoints:
# 1. if map features or traffic signs, it will be "point" always
# 2. if looking for coverage, it will be "image" for points, "sequence" for lines, or "overview" for far zoom
tile_layer = "image"

# Mapillary access token -- user should provide their own
access_token = os.getenv("MAPILLARY_STREET_VIEW_API_KEY")


#       minx       miny        maxx       maxy
#  121.457141  24.960503  121.665934  25.210175     Taipei City
#  118.137972  24.160258  119.479206  24.999617     Kinmen County

# a bounding box in [east_lng,_south_lat,west_lng,north_lat] or [minx, miny, maxx, maxy] format
west, south, east, north = [118.137972, 24.160258, 119.479206, 24.999617]

# get the list of tiles with x and y coordinates which intersect our bounding box
# MUST be at zoom level 14 where the data is available, other zooms currently not supported
tiles = list(mercantile.tiles(west, south, east, north, 14))
# print(tiles)

# # loop through list of tiles to get tile z/x/y to plug in to Mapillary endpoints and make request
for tile in tiles:
    tile_url = 'https://tiles.mapillary.com/maps/vtp/{}/2/{}/{}/{}?access_token={}'.format(tile_coverage, tile.z,
                                                                                           tile.x, tile.y, access_token)
    response = requests.get(tile_url)
    # turn request.content which is "binary" filetype into geojson form
    data = vt_bytes_to_geojson(response.content, tile.x, tile.y, tile.z, layer=tile_layer)

#     # push to output geojson object if yes
    for feature in data['features']:

        # get lng,lat of each feature
        lng = feature['geometry']['coordinates'][0]
        lat = feature['geometry']['coordinates'][1]
        pnt = Point(lng, lat)
#         output['features'].append(feature)

        # ensure feature falls inside bounding box since tiles can extend beyond
        # if lng > west and lng < east and lat > south and lat < north:



                    if polygon.contains(pnt):
                        points.append(pnt)


            # create a folder for each unique sequence ID to group images by sequence
            sequence_id = feature['properties']['sequence_id']
            if not os.path.exists(sequence_id):
                os.makedirs(sequence_id)

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
                      'merge_cc, '
                      'mesh, '
                      'sequence, '
                      'sfm_cluster, '
                      'width, '
                      'detections')

            url = 'https://graph.mapillary.com/{}?fields={}'.format(image_id, fields)
            # url = 'https://graph.mapillary.com/{}?'.format(image_id)
            r = requests.get(url, headers=header)
            data = r.json()
            print(data)
#             image_url = data['thumb_2048_url']
#
#             # save each image with ID as filename to directory by sequence ID
#             with open('{}/{}.jpg'.format(sequence_id, image_id), 'wb') as handler:
#                 image_data = requests.get(image_url, stream=True).content
#                 handler.write(image_data)
#
# # save a local geojson with the filtered data
# with open('sequences.geojson', 'w') as f:
#     json.dump(output, f)
