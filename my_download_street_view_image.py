
# image param
number_images: int = 10_000
path: str = ''

url: str = "https//maps.googleapis.com/maps/api/streetview"
# https://maps.googleapis.com/maps/api/streetview?size=600x300&location=46.414382,10.013988&heading=151.78&pitch=-0.76&key=YOUR_API_KEY&signature=YOUR_SIGNATURE
# street view param

location: tuple[float, float] = (40.457375, -80.009353)  # example
size: tuple[int, int] = (640, 640)
api_key: str = ''

# street view optional param

signature: str = ''
heading: int = 0
headings: list[int] = [0, 90, 180, 270]
fov: int = 90
pitch: int = 0


# generate lat and lng along side in taiwan



#
