import requests
import json
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

print("Querying NASA CMR for the latest IMERG Early Run data...")
cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json?short_name=GPM_3IMERGHHE&page_size=1&sort_key=-start_date"
response = requests.get(cmr_url).json()
granule = response['feed']['entry'][0]

# Extract the NetCDF download link and the timestamp
links = granule['links']
download_url = next(link['href'] for link in links if link['href'].endswith('.nc4'))
start_time = granule['time_start']

print(f"Downloading: {download_url}")
with requests.Session() as s:
    r = s.get(download_url)
    with open("imerg.nc4", "wb") as f:
        f.write(r.content)

print("Processing satellite data into map image...")
ds = xr.open_dataset("imerg.nc4", engine="netcdf4")

# Extract the precipitation data variable
if 'precipitation' in ds:
    precip = ds['precipitation'].squeeze().values
else:
    precip = ds['precipitationCal'].squeeze().values

# Reshape data to fit the map: Transpose to (Lat, Lon) and flip vertically
data = np.flipud(precip.T)

# Apply the weather color map (Blue -> Cyan -> Green -> Yellow -> Red -> Magenta)
cmap = plt.get_cmap('jet')
data_rgba = cmap(plt.Normalize(vmin=0.1, vmax=50)(data))

# Mask out areas with no rain (< 0.1 mm/hr) and empty data (NaN) by making them transparent
data_rgba[data < 0.1, 3] = 0
data_rgba[np.isnan(data), 3] = 0

plt.imsave("imerg_latest.png", data_rgba)

print("Saving timestamp metadata...")
with open("imerg_info.json", "w") as f:
    json.dump({"time": start_time}, f)

print("Update complete!")
