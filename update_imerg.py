import os
import sys
import json
import requests
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg') # Prevents display errors in headless servers
import matplotlib.pyplot as plt

# --- FIX FOR GITHUB ACTIONS IPv6 BUG WITH NASA ---
import socket
import urllib3.util.connection as urllib3_cn
urllib3_cn.allowed_gai_family = lambda: socket.AF_INET
# -------------------------------------------------

user = os.environ.get('EARTHDATA_USER')
password = os.environ.get('EARTHDATA_PASS')

print("1. Querying NASA CMR for the latest IMERG Early Run data...")
cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json?short_name=GPM_3IMERGHHE&page_size=1&sort_key=-start_date"
response = requests.get(cmr_url)

if not response.ok:
    print(f"Failed to reach NASA CMR: {response.text}")
    sys.exit(1)

data = response.json()
granule = data['feed']['entry'][0]

# Extract the download link
links = granule['links']
download_url = None
for link in links:
    href = link.get('href', '')
    if href.endswith('.HDF5') or href.endswith('.nc4') or href.endswith('.h5'):
        download_url = href
        break

if not download_url:
    print("Error: Could not find a valid data download link.")
    print("Available links:", [l.get('href') for l in links])
    sys.exit(1)

start_time = granule['time_start']

print(f"2. Downloading: {download_url}")

# NASA uses a redirect system for logins.
with requests.Session() as s:
    s.auth = (user, password)
    r1 = s.request('get', download_url)
    r = s.get(r1.url, auth=(user, password))
    
    if r.ok:
        with open("imerg_data.hdf5", "wb") as f:
            f.write(r.content)
        print("Download successful!")
    else:
        print(f"Download failed with status code {r.status_code}")
        print("Check if you approved 'NASA GES DISC DATA ARCHIVE' in your Earthdata account.")
        sys.exit(1)

print("3. Processing satellite data into map image...")
try:
    # FIX: NASA V07 hides the data inside a group called 'Grid'
    ds = xr.open_dataset("imerg_data.hdf5", engine="h5netcdf", group="Grid")
except Exception:
    # Fallback to root just in case they revert it
    try:
        ds = xr.open_dataset("imerg_data.hdf5", engine="h5netcdf")
    except Exception as e:
        print(f"Error opening file. Details: {e}")
        sys.exit(1)

# Extract precipitation
if 'precipitation' in ds:
    precip = ds['precipitation'].squeeze().values
elif 'precipitationCal' in ds:
    precip = ds['precipitationCal'].squeeze().values
else:
    print(f"Error: Could not find precipitation variable. Available variables: {list(ds.keys())}")
    sys.exit(1)

data_val = np.flipud(precip.T)

# Paint the map
cmap = plt.get_cmap('jet')
data_rgba = cmap(plt.Normalize(vmin=0.1, vmax=50)(data_val))

# Make empty areas transparent
data_rgba[data_val < 0.1, 3] = 0
data_rgba[np.isnan(data_val), 3] = 0

plt.imsave("imerg_latest.png", data_rgba)

print("4. Saving timestamp metadata...")
with open("imerg_info.json", "w") as f:
    json.dump({"time": start_time}, f)

print("Update complete!")
