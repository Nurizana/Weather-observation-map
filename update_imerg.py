import os
import sys
import json
import requests
import xarray as xr
import numpy as np
import time  # Added for delay between downloads
import matplotlib
matplotlib.use('Agg') # Prevents display errors in headless servers
import matplotlib.pyplot as plt

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- FIX FOR GITHUB ACTIONS IPv6 BUG WITH NASA ---
import socket
import urllib3.util.connection as urllib3_cn
urllib3_cn.allowed_gai_family = lambda: socket.AF_INET
# -------------------------------------------------

user = os.environ.get('EARTHDATA_USER')
password = os.environ.get('EARTHDATA_PASS')

print("1. Querying NASA CMR for the 10 latest IMERG Early Run data...")
# Requesting 10 granules instead of 1
cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json?short_name=GPM_3IMERGHHE&page_size=10&sort_key=-start_date"
response = requests.get(cmr_url)

if not response.ok:
    print(f"Failed to reach NASA CMR: {response.text}")
    sys.exit(1)

data = response.json()
entries = data['feed']['entry']

if not entries:
    print("Error: No data granules found.")
    sys.exit(1)

# Reverse the entries so the oldest is first (index 0) and newest is last (index 9)
# This ensures the animation plays chronologically forward in time.
entries.reverse()

frames_data = []

# --- SETUP RETRY STRATEGY & SESSION ---
# This will retry up to 5 times for common server-side errors/timeouts, 
# increasing the delay between retries automatically.
retry_strategy = Retry(
    total=5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"],
    backoff_factor=1
)
adapter = HTTPAdapter(max_retries=retry_strategy)

with requests.Session() as s:
    s.auth = (user, password)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    
    for i, granule in enumerate(entries):
        start_time = granule['time_start']
        
        # Extract the download link
        links = granule['links']
        download_url = None
        for link in links:
            href = link.get('href', '')
            if href.endswith('.HDF5') or href.endswith('.nc4') or href.endswith('.h5'):
                download_url = href
                break

        if not download_url:
            print(f"Warning: Could not find a valid data download link for granule {i}. Skipping.")
            continue

        print(f"\n--- Processing Frame {i+1}/10 ---")
        print(f"Time: {start_time}")
        print(f"Downloading: {download_url}")

        try:
            # ADDED: 30-second timeout to prevent infinite hanging connections
            r1 = s.request('get', download_url, timeout=30)
            r = s.get(r1.url, auth=(user, password), timeout=30)
            
            if r.ok:
                with open("imerg_data.hdf5", "wb") as f:
                    f.write(r.content)
                print("Download successful. Processing image...")
            else:
                print(f"Download failed with status code {r.status_code}. Skipping.")
                continue
                
        except requests.exceptions.RequestException as e:
            print(f"Network error during download: {e}. Skipping.")
            continue

        try:
            # FIX: NASA V07 hides the data inside a group called 'Grid'
            ds = xr.open_dataset("imerg_data.hdf5", engine="h5netcdf", group="Grid")
        except Exception:
            # Fallback to root just in case they revert it
            try:
                ds = xr.open_dataset("imerg_data.hdf5", engine="h5netcdf")
            except Exception as e:
                print(f"Error opening file. Details: {e}. Skipping.")
                continue

        # Extract precipitation
        if 'precipitation' in ds:
            precip = ds['precipitation'].squeeze().values
        elif 'precipitationCal' in ds:
            precip = ds['precipitationCal'].squeeze().values
        else:
            print(f"Warning: Could not find precipitation variable. Skipping.")
            ds.close()
            continue

        data_val = np.flipud(precip.T)

        # Paint the map
        cmap = plt.get_cmap('jet')
        data_rgba = cmap(plt.Normalize(vmin=0.1, vmax=50)(data_val))

        # Make empty areas transparent
        data_rgba[data_val < 0.1, 3] = 0
        data_rgba[np.isnan(data_val), 3] = 0
        
        ds.close() # Free up resources for the next iteration

        # Save indexed image
        image_filename = f"imerg_{i}.png"
        plt.imsave(image_filename, data_rgba)
        print(f"Saved {image_filename}")

        # Append to metadata array
        frames_data.append({
            "time": start_time,
            "image": image_filename
        })
        
        # ADDED: 2-second sleep to avoid hitting NASA's rate limits/connection blocks
        time.sleep(2)

print("\n4. Saving timestamp metadata...")
with open("imerg_info.json", "w") as f:
    # Save the array as JSON
    json.dump(frames_data, f, indent=4)

print("Update complete!")
