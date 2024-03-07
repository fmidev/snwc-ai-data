import xarray as xr
import sys
import s3fs
import numpy as np


def open_file(filename, store=None):
    if filename.endswith(".zarr"):
        if store is None:
            store = filename
        return xr.open_zarr(store=store)
    elif filename.endswith(".nc"):
        return xr.open_dataset(filename)
    else:
        raise ValueError("Unknown file format")


if len(sys.argv) < 2:
    print("Usage: read-xarray.py <filename>")
    sys.exit(1)

filename = sys.argv[1]

if filename.startswith("s3://"):
    s3 = s3fs.S3FileSystem(anon=True, endpoint_url="https://lake.fmi.fi")
    ds = open_file(filename, s3)
else:
    ds = open_file(filename)

print(ds)
print(ds.spatial_ref)
print(ds.variables["time"])
for var in ds.variables:
    arr = ds.variables[var].to_numpy()
    if arr.dtype not in (np.dtype("datetime64[ns]"),):
        print(
            "{} shape={} min={:.3f} mean={:.3f} max={:.3f}".format(
                var, arr.shape, np.min(arr), np.mean(arr), np.max(arr)
            )
        )
