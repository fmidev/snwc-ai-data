import numpy as np
import eccodes as ecc
import argparse
import datetime
import fsspec
import os
import zarr
import xarray as xr
import rioxarray
import rasterio
import cartopy
from rasterio.session import AWSSession
from tqdm import tqdm
from scipy.interpolate import interpn
from botocore.exceptions import ClientError

points = None
interp_points = None
interp_points_shape = None
x = None
y = None
CACHE = {}
CACHEFILE = []


def create_default_params(args):
    args.parameters = []
    for l in (300, 500, 700, 850, 925, 1000):
        for p in ("z", "u", "v"):
            args.parameters.append("{}_isobaricInhPa_{}".format(p, l))

    args.parameters += ["pres_heightAboveSea_0"]

    args.parameters.sort()

    assert len(args.parameters) == 19

    # print("Using default param list: {}".format(args.params))


def parse_date(datestr):
    try:
        return datetime.datetime.strptime(datestr, "%Y%m%d")
    except ValueError:
        return datetime.datetime.strptime(datestr, "%Y%m%d%H")


def parse_size(sizestr):
    return tuple(map(int, sizestr.split("x")))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create zarr archive from meps and nwcsaf data"
    )
    parser.add_argument("outfile", type=str, help="output file name")
    parser.add_argument(
        "--parameters",
        type=str,
        help="comma separated list of parameters to interpolate",
    )
    parser.add_argument(
        "--input-size",
        type=parse_size,
        default=(224, 224),
        help="size of target grid, width_x_height. Default: 224x224",
    )
    parser.add_argument(
        "--start-datetime",
        type=parse_date,
        required=True,
        help="Start date ({yyyymmdd | yyyymmddhh})",
    )
    parser.add_argument(
        "--stop-datetime",
        type=parse_date,
        required=True,
        help="Stop date ({yyyymmdd | yyyymmddhh})",
    )
    parser.add_argument(
        "--subhourly",
        action="store_true",
        help="Interpolate data to subhourly resolution",
        default=False,
    )

    args = parser.parse_args()

    if args.parameters is None:
        create_default_params(args)
    else:
        args.parameters = args.parameters.split(",")

    return args


args = parse_args()


def read_static_data():
    data = []
    dt = datetime.datetime(2021, 4, 1, 0, 0)
    for param in ("lsm", "z"):
        filename = "meps-ai-data/meps/const/{}_heightAboveGround_0.grib2".format(param)
        read_to_cache(filename, param)

        data.append(get_from_cache(dt, param))

    return data[0], data[1]


def read_nwcsaf_data(orig_dates):
    filenames = [
        "s3://cc_archive/nwcsaf/{}/{}_nwcsaf_effective-cloudiness.grib2".format(
            x.strftime("%Y/%m/%d"), x.strftime("%Y%m%dT%H%M%S")
        )
        for x in orig_dates
    ]
    data = []
    dates = []
    nx = None
    ny = None

    for i, filename in enumerate(filenames):
        uri = "simplecache::{}".format(filename)
        try:
            file_obj = fsspec.open_local(
                uri,
                mode="rb",
                s3={
                    "anon": True,
                    "client_kwargs": {"endpoint_url": "https://lake.fmi.fi"},
                },
            )
        except ClientError as e:
            print("Failed to open file: {}".format(filename))
            continue
        except FileNotFoundError as e:
            print("Failed to open file: {}".format(filename))
            continue

        with open(file_obj, "rb") as f:
            paramdata = []
            gh = ecc.codes_grib_new_from_file(f)

            assert gh is not None, "Failed to read file {}".format(filename)

            if nx is None:
                nx = ecc.codes_get(gh, "Ni")
                ny = ecc.codes_get(gh, "Nj")

            values = ecc.codes_get_values(gh).astype(np.float32)
            values = values.reshape((ny, nx))
            values = np.flipud(values)
            ecc.codes_release(gh)

            if values.shape != interp_points_shape:
                values = interpn(points, values, interp_points).reshape(
                    interp_points_shape
                )

            data.append(values)
            dates.append(orig_dates[i])

        os.remove(file_obj)
        os.rmdir(os.path.dirname(file_obj))

    data = np.expand_dims(np.asarray(data), axis=1).astype(np.float32)
    return dates, data


def read_to_cache(filename, param):
    global CACHE
    global CACHEFILE

    if filename in CACHEFILE:
        return

    uri = "simplecache::{}".format("s3://" + filename)
    file_obj = fsspec.open_local(
        uri,
        mode="rb",
        s3={"anon": True, "client_kwargs": {"endpoint_url": "https://lake.fmi.fi"}},
    )

    nx = None
    ny = None

    with open(file_obj, "rb") as f:
        while True:
            try:
                gh = ecc.codes_grib_new_from_file(f)
            except ecc.CodesInternalError as e:
                print("ERROR reading file: {}".format(filename))
                raise ValueError(e)

            if gh is None:
                break

            if nx is None:
                nx = ecc.codes_get(gh, "Ni")
                ny = ecc.codes_get(gh, "Nj")

            dataDate = ecc.codes_get(gh, "dataDate")
            dataTime = int(ecc.codes_get(gh, "dataTime"))
            date = datetime.datetime.strptime(
                str(dataDate) + "{:04d}".format(dataTime), "%Y%m%d%H%M"
            )
            values = ecc.codes_get_values(gh).astype(np.float32)
            values = values.reshape((ny, nx))
            values = np.flipud(values)
            ecc.codes_release(gh)

            values = (
                interpn(points, values, interp_points)
                .reshape(interp_points_shape)
                .astype(np.float32)
            )

            CACHE[(date, param)] = values

    os.remove(file_obj)
    os.rmdir(os.path.dirname(file_obj))

    CACHEFILE.append(filename)


def get_from_cache(date, param):
    global CACHE
    return CACHE.get((date, param), None)


def read_meps_data(daterange):
    data = []
    for param in args.parameters:
        paramdata = []
        for datetime in daterange:
            ss = time.time()
            filename = generate_filename(datetime, param)
            read_to_cache(filename, param)
            paramdata.append(get_from_cache(datetime, param))

        data.append(paramdata)

    data = np.asarray(data).astype(np.float32).swapaxes(0, 1)

    assert data.shape == (
        len(daterange),
        len(args.parameters),
        interp_points_shape[0],
        interp_points_shape[1],
    ), "Shape is {}, should be: {}".format(
        data.shape,
        (
            len(daterange),
            len(args.parameters),
            interp_points_shape[0],
            interp_points_shape[1],
        ),
    )

    return data


def interpolate_to_subhourly(dates, data):
    new_data = []
    new_dates = []
    for i in range(len(dates) - 1):
        new_dates.append(dates[i])
        new_data.append(data[i])
        for j, fraction in enumerate((0.25, 0.5, 0.75)):
            new_dates.append(dates[i] + datetime.timedelta(minutes=(j + 1) * 15))
            new_data.append(data[i] + fraction * (data[i + 1] - data[i]))

    new_dates.append(dates[-1])
    new_data.append(data[-1])

    return new_dates, np.asarray(new_data).astype(np.float32)


def create_xarray_dataset(data, dates):
    if len(dates) == 0:
        return None

    proj = create_spatial_ref()
    lat, lon = create_latlon_grid(proj, x, y)
    ds = xr.Dataset(
        coords={
            "x": x,
            "y": y,
            "longitude": (["y", "x"], lon),
            "latitude": (["y", "x"], lat),
            "time": dates,
        },
    )

    assert data.dtype == np.float32

    lsm, z = read_static_data()

    ds["lsm_heightAboveGround_0"] = (["y", "x"], lsm)
    ds["z_heightAboveGround_0"] = (["y", "x"], z)

    for i, param in enumerate(
        args.parameters + ["effective_cloudiness_heightAboveGround_0"]
    ):
        ds[param] = (["time", "y", "x"], data[:, i, :, :])

    ds = ds.drop_duplicates(dim="time", keep="first")

    # remove the last time (=args.stop_datetime); out timerange is
    # start-inclusive and stop-exclusive

    try:
        stop_index = ds.coords["time"].to_index().get_loc(args.stop_datetime)
        ds = ds.isel(time=slice(None, stop_index))
    except KeyError:
        pass

    ds = ds.chunk(
        {
            "time": 2,
            "y": interp_points_shape[0],
            "x": interp_points_shape[1],
        }
    )

    ds.rio.write_crs(create_spatial_ref(), inplace=True)

    return ds


def merge(dates1, data1, dates2, data2):
    _data1 = []
    _data2 = []
    dates = []
    for i, date in enumerate(dates1):
        if date not in dates2:
            print("Date {} not found in nwcsaf data".format(date))
            continue

        dates.append(date)
        _data1.append(data1[i])
        _data2.append(data2[dates2.index(date)])

    if len(dates) == 0:
        print("No valid data found")
        return [], []

    _data1 = np.asarray(_data1)
    _data2 = np.asarray(_data2)
    assert (
        _data1.shape[0] == _data2.shape[0]
    ), "Length of data does not match: {} vs {}".format(data1.shape, data2.shape)

    data = np.concatenate((_data1, _data2), axis=1)

    return dates, data


def generate_filename(datetime, param):
    return "s3://meps-ai-data/meps/{}/{}/{}/{}_{}.grib2".format(
        datetime.strftime("%Y"),
        datetime.strftime("%m"),
        datetime.strftime("%d"),
        datetime.strftime("%Y%m%d"),
        param,
    )


def create_spatial_ref():
    lambert_proj_params = {
        "a": 6371229,
        "b": 6371229,
        "lat_0": 63.3,
        "lat_1": 63.3,
        "lat_2": 63.3,
        "lon_0": 15.0,
        "proj": "lcc",
    }

    # Create projection
    proj = cartopy.crs.LambertConformal(
        central_longitude=lambert_proj_params["lon_0"],
        central_latitude=lambert_proj_params["lat_0"],
        standard_parallels=(lambert_proj_params["lat_1"], lambert_proj_params["lat_2"]),
        globe=cartopy.crs.Globe(
            ellipse="sphere",
            semimajor_axis=lambert_proj_params["a"],
            semiminor_axis=lambert_proj_params["b"],
        ),
    )

    return proj


def create_latlon_grid(proj, x, y):
    lon, lat = np.meshgrid(x, y)

    pc = cartopy.crs.PlateCarree()
    for i in range(lon.shape[0]):
        for j in range(lat.shape[1]):
            lon[i, j], lat[i, j] = pc.transform_point(
                lon[i, j], lat[i, j], src_crs=proj
            )

    return lat, lon


def create_meps_grid():
    proj = create_spatial_ref()
    nx = 949
    ny = 1069

    bottom_left = (0.27828, 50.3196)
    grid_size = (ny, nx)
    NX = 2500 * (nx - 1)
    NY = 2500 * (ny - 1)

    print(
        "MEPS grid size: {} / {:.0f}x{:.0f}m".format(
            grid_size, NX / (nx - 1), NY / (ny - 1)
        )
    )

    x0, y0 = proj.transform_point(
        bottom_left[0], bottom_left[1], src_crs=cartopy.crs.PlateCarree()
    )

    x = np.linspace(x0, x0 + NX, grid_size[1])
    y = np.linspace(y0, y0 + NY, grid_size[0])

    return y, x


def create_proj_grid():
    proj = create_spatial_ref()
    nx = args.input_size[0]
    ny = args.input_size[1]

    bottom_left = (0.27828, 50.3196)
    grid_size = (ny, nx)
    NX = 2500 * (949 - 1)  # original MEPS grid length
    NY = 2500 * (1069 - 1)

    dx = NX / (nx - 1)
    dy = NY / (ny - 1)

    print("Interpolation grid size: {} / {:.0f}x{:.0f}m".format(grid_size, dx, dy))

    x0, y0 = proj.transform_point(
        bottom_left[0], bottom_left[1], src_crs=cartopy.crs.PlateCarree()
    )

    x = np.linspace(x0, x0 + NX, grid_size[1])
    y = np.linspace(y0, y0 + NY, grid_size[0])

    return y, x


def save_ds(filename, ds):
    if ds is None:
        return
    if os.path.exists(filename):
        ds.to_zarr(filename, append_dim="time")
    else:
        ds.to_zarr(filename)


def create_interpolation_grid():
    global points, interp_points, interp_points_shape, x, y
    points = create_meps_grid()  # y, x
    interp_points = create_proj_grid()  # y, x

    interp_points_shape = (interp_points[0].shape[0], interp_points[1].shape[0])
    x = interp_points[1]
    y = interp_points[0]
    X, Y = np.meshgrid(x, y)
    interp_points = np.concatenate((Y.reshape(-1, 1), X.reshape(-1, 1)), axis=1)


def main():
    if os.path.exists(args.outfile):
        print("File {} already exists, aborting".format(args.outfile))
        return

    create_interpolation_grid()

    daterange = [
        args.start_datetime + datetime.timedelta(seconds=x)
        for x in range(
            0,
            int(3600 + (args.stop_datetime - args.start_datetime).total_seconds()),
            3600,
        )
    ]

    for i in tqdm(range(24, len(daterange), 24)):
        prev = daterange[i - 24]
        cur = daterange[i]

        dates1 = [
            prev + datetime.timedelta(seconds=x) for x in range(0, 3600 * 25, 3600)
        ]
        data1 = read_meps_data(dates1)
        assert data1.dtype == np.float32

        if args.subhourly:
            dates1, data1 = interpolate_to_subhourly(dates1, data1)

        dates2, data2 = read_nwcsaf_data(dates1)
        assert data2.dtype == np.float32

        # remove "stop time" value from dataset ; our date range is stop exclusive

        dates, data = merge(dates1, data1, dates2, data2)
        try:
            last_index = dates.index(cur)
            dates = dates[:last_index]
            data = data[:last_index]
        except KeyError:
            pass
        except ValueError:
            pass

        ds = create_xarray_dataset(data, dates)
        save_ds(args.outfile, ds)

    root = xr.open_zarr(args.outfile)
    print(root)
    # print("Wrote data of shape {} to file {}".format(root["data"].shape, filename))


main()
