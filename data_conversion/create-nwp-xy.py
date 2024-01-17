#!/usr/bin/env python3

import numpy as np
from osgeo import gdal, osr
import sys
import argparse

parser = argparse.ArgumentParser(
    description="Create a 2d MEPS2500D grid with given nx ny numbers"
)
parser.add_argument(
    "--nx", type=int, help="number of grid points in x direction", default=238
)
parser.add_argument(
    "--ny", type=int, help="number of grid points in y direction", default=268
)
parser.add_argument("--output", type=str, help="output file", required=True)
args = parser.parse_args()


def xy_from_latlon(lon, lat):
    src = osr.SpatialReference()
    tgt = osr.SpatialReference()
    src.ImportFromEPSG(4326)
    tgt.ImportFromProj4(proj4)

    transform = osr.CoordinateTransformation(src, tgt)
    x, y, _ = transform.TransformPoint(lat, lon)
    return x, y


# lamber conformal conic projection
proj4 = "+proj=lcc +lat_1=63.3 +lat_2=63.3 +lat_0=63.3 +lon_0=15.0 +R=6371229 +units=m +no_defs"

# grid size
o_nx = 949
o_ny = 1069

# grid spacing
o_dx = 2500
o_dy = 2500

# area size
o_size_x = (o_nx - 1) * o_dx
o_size_y = (o_ny - 1) * o_dy

n_dx = o_size_x / (args.nx - 1.0)
n_dy = o_size_y / (args.ny - 1.0)

print(f"New grid size: {args.nx} x {args.ny} spacing {n_dx} x {n_dy}")
x = []
y = []

x0, y0 = xy_from_latlon(0.27828, 50.319616)

for i in range(args.nx):
    x.append(x0 + i * n_dx)

for i in range(args.ny):
    y.append(y0 + i * n_dy)

x = np.asarray(x)
y = np.asarray(y)

xv, yv = np.meshgrid(x, y)  # np.asarray([x, y])

arr = np.stack((xv, yv), axis=0).astype(np.float32)

print(arr.shape)

np.save(args.output, arr)

print("Saved to", args.output)
