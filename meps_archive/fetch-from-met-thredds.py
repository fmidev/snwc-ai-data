#!/usr/bin/env python3

import argparse
import sys
import subprocess
import netCDF4 as nc
import eccodes as ecc
import os
import numpy as np
import datetime
import pytz
import random
from nco import Nco

coalesce_times = True


def parse_args():
    parser = argparse.ArgumentParser("Read from arcus", add_help=False)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--month", required=True, type=int)
    parser.add_argument("--day", required=True, type=int)
    parser.add_argument("--cycle", required=True, type=int)
    parser.add_argument("--leadtimes", required=True, type=str)
    parser.add_argument("--param", required=True, type=str)
    parser.add_argument("--level", required=True, type=str)
    parser.add_argument("--level-values", required=True, type=str)
    parser.add_argument("--output-dir", required=False, type=str, default="/tmp")
    parser.add_argument("--output-filename", required=False, type=str, default=None)
    parser.add_argument(
        "--merge",
        action="store_true",
        default=False,
        help="Merge all grib fields into one file",
    )
    parser.add_argument("--perturbation-number", default=0, type=int)
    parser.add_argument("--use-deterministic-file", action="store_true", default=False)
    args = parser.parse_args()

    args.leadtimes = list(map(lambda x: int(x), args.leadtimes.split(",")))
    args.leadtimes.sort()
    args.level_values = list(map(lambda x: int(x), args.level_values.split(",")))

    if args.merge and args.output_filename is None:
        print("Output filename must be specified when merging files")
        sys.exit(1)

    global coalesce_times

    for i in range(1, len(args.leadtimes)):
        if args.leadtimes[i] - args.leadtimes[i - 1] > 1:
            coalesce_times = False
            break

    return args


args = parse_args()


def pv():
    return np.array(
        [
            0.00000000e00,
            2.00000000e03,
            4.00021289e03,
            6.00209668e03,
            7.91125830e03,
            9.63301074e03,
            1.11693711e04,
            1.25225771e04,
            1.36950020e04,
            1.46891152e04,
            1.55074902e04,
            1.61546973e04,
            1.66321250e04,
            1.69401504e04,
            1.70823496e04,
            1.70652812e04,
            1.68981836e04,
            1.65925898e04,
            1.61619043e04,
            1.56209434e04,
            1.49854648e04,
            1.42717080e04,
            1.34959600e04,
            1.26741689e04,
            1.18216035e04,
            1.09525703e04,
            1.00802002e04,
            9.21628613e03,
            8.37117871e03,
            7.55374463e03,
            6.77135449e03,
            6.02992041e03,
            5.33395898e03,
            4.68668066e03,
            4.09009521e03,
            3.54512646e03,
            3.05173804e03,
            2.60905811e03,
            2.21550464e03,
            1.86890771e03,
            1.56662817e03,
            1.30566882e03,
            1.08185498e03,
            8.90475952e02,
            7.27745483e02,
            5.90177490e02,
            4.74587677e02,
            3.78088562e02,
            2.98079468e02,
            2.32233124e02,
            1.78480148e02,
            1.34992081e02,
            1.00163689e02,
            7.25952911e01,
            5.10750885e01,
            3.45621643e01,
            2.21702213e01,
            1.31522598e01,
            6.88641310e00,
            2.86306143e00,
            6.73443556e-01,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            9.54680028e-04,
            3.82569991e-03,
            8.62327032e-03,
            1.53578203e-02,
            2.40404606e-02,
            3.46831419e-02,
            4.72983904e-02,
            6.19510189e-02,
            7.86818713e-02,
            9.74432528e-02,
            1.18155859e-01,
            1.40710980e-01,
            1.64973482e-01,
            1.90785542e-01,
            2.17970863e-01,
            2.46339247e-01,
            2.75691181e-01,
            3.05822432e-01,
            3.36528242e-01,
            3.67607266e-01,
            3.98864776e-01,
            4.30115640e-01,
            4.61186230e-01,
            4.91916239e-01,
            5.22159457e-01,
            5.51784456e-01,
            5.80674410e-01,
            6.08727098e-01,
            6.35853887e-01,
            6.61979139e-01,
            6.87038958e-01,
            7.10980356e-01,
            7.33759642e-01,
            7.55341411e-01,
            7.75697351e-01,
            7.94804871e-01,
            8.12645972e-01,
            8.29206347e-01,
            8.44540000e-01,
            8.58755052e-01,
            8.71918023e-01,
            8.84092748e-01,
            8.95340443e-01,
            9.05719638e-01,
            9.15286422e-01,
            9.24094498e-01,
            9.32195485e-01,
            9.39638972e-01,
            9.46472764e-01,
            9.52743292e-01,
            9.58495498e-01,
            9.63773429e-01,
            9.68620062e-01,
            9.73078012e-01,
            9.77189422e-01,
            9.80996370e-01,
            9.84541297e-01,
            9.87867296e-01,
            9.91024613e-01,
            9.94065106e-01,
            9.97039258e-01,
            1.00000000e00,
        ]
    )


def param_to_id(param):
    # product definition template number
    # discipline
    # parameter category
    # parameter number
    # type of statistical processing
    params = {
        "air_temperature_0m": (1, 0, 0, 0, None),
        "air_temperature_2m": (1, 0, 0, 0, None),
        "air_temperature_pl": (1, 0, 0, 0, None),
        "geopotential_pl": (1, 0, 3, 4, None),
        "surface_geopotential": (1, 0, 3, 4, None),
        "relative_humidity_2m": (1, 0, 1, 192, None),
        "relative_humidity_pl": (1, 0, 1, 192, None),
        "x_wind_10m": (1, 0, 2, 2, None),
        "x_wind_pl": (1, 0, 2, 2, None),
        "y_wind_10m": (1, 0, 2, 3, None),
        "y_wind_pl": (1, 0, 2, 3, None),
        "atmosphere_boundary_layer_thickness": (1, 0, 19, 3, None),
        "wind_speed_of_gust": (11, 0, 2, 22, 2),
        "air_pressure_at_sea_level": (1, 0, 3, 0, None),
        "surface_air_pressure": (1, 0, 3, 0, None),
        "specific_humidity_pl": (1, 0, 1, 0, None),
        "upward_air_velocity_pl": (1, 0, 2, 9, None),
        "turbulent_kinetic_energy_pl": (1, 0, 19, 11, None),
        "cloud_top_altitude": (1, 0, 6, 12, None),
        "cloud_base_altitude": (1, 0, 6, 11, None),
        "precipitation_amount_acc": (11, 0, 1, 8, 0),
        "mass_fraction_of_cloud_condensed_water_in_air_pl": (1, 0, 1, 83, None),
        "mass_fraction_of_cloud_ice_in_air_pl": (1, 0, 1, 84, None),
        "mass_fraction_of_rain_in_air_pl": (1, 0, 1, 85, None),
        "mass_fraction_of_snow_in_air_pl": (1, 0, 1, 86, None),
        "mass_fraction_of_graupel_in_air_pl": (1, 0, 1, 32, None),
        "cloud_area_fraction": (1, 0, 6, 32, None),
        "high_type_cloud_area_fraction": (1, 0, 6, 196, None),
        "medium_type_cloud_area_fraction": (1, 0, 6, 195, None),
        "low_type_cloud_area_fraction": (1, 0, 6, 194, None),
        "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time": (
            11,
            0,
            4,
            3,
            0,
        ),
        "integral_of_surface_downwelling_longwave_flux_in_air_wrt_time": (
            11,
            0,
            5,
            4,
            0,
        ),
        "specific_convective_available_potential_energy": (1, 0, 7, 6, None),
        "visibility_in_air": (1, 0, 19, 0, None),
        "precipitation_type": (1, 0, 1, 19, None),
        "lightning_index": (1, 0, 17, 192, None),
    }

    return params[param]


def level_type_to_id(level_type):
    levels = {
        "pressure": 100,
        "height_above_msl": 102,
        "height0": 103,
        "height1": 103,
        "height2": 103,
        "height6": 103,
        "height7": 103,
        "surface": 103,
    }

    return levels[level_type]


def level_value_to_index(level, level_value):
    if level == "pressure":
        if args.use_deterministic_file:
            # 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 700.0, 800.0, 850.0, 925.0, 1000.0
            idx = [50, 100, 150, 200, 250, 300, 400, 500, 700, 800, 850, 925, 1000]
        else:
            idx = [300, 500, 700, 850, 925, 1000]

        return idx.index(level_value)

    return 0  # all other levels which are one-height levels


def bits_per_value(param):
    return 16


def common_param_name(param):
    if param in ("air_temperature_pl", "air_temperature_2m", "air_temperature_0m"):
        return "t"
    if param in ["surface_geopotential", "geopotential_pl"]:
        return "z"
    if param in ("relative_humidity_pl", "relative_humidity_2m"):
        return "r"
    if param in ("x_wind_pl", "x_wind_10m"):
        return "u"
    if param in ("y_wind_pl", "y_wind_10m"):
        return "v"
    if param == "atmosphere_boundary_layer_thickness":
        return "mld"
    if param == "wind_speed_of_gust":
        return "fg"
    if param in ("surface_air_pressure", "air_pressure_at_sea_level"):
        return "pres"
    if param in ("specific_humidity_pl", "specific_humidity_2m"):
        return "q"
    if param == "cloud_top_altitude":
        return "cdct"
    if param == "cloud_base_altitude":
        return "cdcb"
    if param == "precipitation_amount_acc":
        return "tp"
    if param == "upward_air_velocity_pl":
        return "w"
    if param == "snowfall_amount_acc":
        return "sf"
    if param == "turbulent_kinetic_energy_pl":
        return "tke"
    if param == "mass_fraction_of_cloud_condensed_water_in_air_pl":
        return "cldwat"
    if param == "mass_fraction_of_cloud_ice_in_air_pl":
        return "cldice"
    if param == "mass_fraction_of_rain_in_air_pl":
        return "rainmr"
    if param == "mass_fraction_of_snow_in_air_pl":
        return "snowmr"
    if param == "mass_fraction_of_graupel_in_air_pl":
        return "graupelmr"
    if param == "cloud_area_fraction":
        return "tcc"
    if param == "high_type_cloud_area_fraction":
        return "hcc"
    if param == "medium_type_cloud_area_fraction":
        return "mcc"
    if param == "low_type_cloud_area_fraction":
        return "lcc"
    if param == "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time":
        return "ssrd"
    if param == "integral_of_surface_downwelling_longwave_flux_in_air_wrt_time":
        return "strd"
    if param == "specific_convective_available_potential_energy":
        return "cape"
    if param == "visibility_in_air":
        return "vis"
    if param == "precipitation_type":
        return "ptype"
    if param == "lightning_index":
        return "li"


def common_level_name(level):
    if level == "pressure":
        return "isobaricInhPa"
    if level in ("height0", "height1", "height2", "height6", "height7", "surface"):
        return "heightAboveGround"
    if level == "height_above_msl":
        return "heightAboveSea"


def get_dodsname():

    dodsname = "meps_lagged_6_h_subset_2_5km"
    no_member = False

    if args.year == 2020 and (args.month == 1 or (args.month == 2 and args.day < 5)):
        raise Exception(
            "Data at this data is not in MEPS2500D domain; try to start with 2020-02-05"
        )

    if args.use_deterministic_file or (
        args.perturbation_number == 0
        and (
            args.year > 2022
            or (
                args.year == 2022
                and (args.month > 8 or (args.month == 8 and args.day >= 7))
            )
        )
    ):
        dodsname = "meps_det_2_5km"
        no_member = True

        return dodsname, no_member

    if args.use_deterministic_file or (
        args.perturbation_number == 0 and args.year >= 2024
    ):
        dodsname = "meps_det_pl" if args.level == "pressure" else "meps_det_sfc"
        no_member = True

    return dodsname, no_member


def create_url():
    ni = 948
    nj = 1068

    archive = "meps25epsarchive"

    dodsname, no_member = get_dodsname()

    # [start:stride:stop]
    coord = (
        "x[0:1:{}],y[0:1:{}],longitude[0:1:{}][0:1:{}],latitude[0:1:{}][0:1:{}]".format(
            ni, nj, nj, ni, nj, ni
        )
    )

    cycle = args.cycle
    member = args.perturbation_number

    if coalesce_times:
        print("Coalescing times {} to single request".format(args.leadtimes))

    def get_time():
        if coalesce_times:
            b = args.leadtimes[0]
            e = args.leadtimes[-1]
            yield "time[{}:1:{}]".format(b, e), b, e

        else:
            for lt in args.leadtimes:
                yield "time[{}:1:{}]".format(lt, lt), lt, lt

    filetype = "nc" if args.year < 2024 else "nc"
    for time, lt_b, lt_e in get_time():
        # for i, lt in enumerate(args.leadtimes):
        # time = "time[{}:1:{}]".format(lt, lt)
        for i, level_value in enumerate(args.level_values):
            lev_index = level_value_to_index(args.level, level_value)
            lev = "{}[{}:1:{}]".format(args.level, lev_index, lev_index)

            if no_member:
                # time, level, nj, ni
                par = "{}[{}:1:{}][{}:1:{}][0:1:{}][0:1:{}]".format(
                    args.param, lt_b, lt_e, lev_index, lev_index, nj, ni
                )
                memb = ""

            else:
                # time, level, member, nj, ni
                par = "{}[{}:1:{}][{}:1:{}][{}:1:{}][0:1:{}][0:1:{}]".format(
                    args.param, lt_b, lt_e, lev_index, lev_index, member, member, nj, ni
                )
                memb = ",ensemble_member[{}:1:{}]".format(member, member)

            url = "https://thredds.met.no/thredds/dodsC/{}/{}/{:02d}/{:02d}/{}_{}{:02d}{:02d}T{:02d}Z.{}?forecast_reference_time,projection_lambert,{},{}{},{},{}".format(
                archive,
                args.year,
                args.month,
                args.day,
                dodsname,
                args.year,
                args.month,
                args.day,
                cycle,
                filetype,
                coord,
                lev,
                memb,
                time,
                par,
            )

            yield url


def fetch_from_thredds():
    datas = []
    for url in create_url():
        print(url)

        tmpfile = "/tmp/meps-{}.nc4".format(int(random.random() * 10000000))

        try:
            os.remove(tmpfile)
        except FileNotFoundError as e:
            pass

        nco = Nco()
        nco.ncks(input=url, output=tmpfile)  # , options=options)

        datas.append(nc.Dataset(tmpfile, "r"))
        print("Read data from {}".format(url))

        os.remove(tmpfile)

    return datas


def convert(datas):

    for i, ds in enumerate(datas):
        print("{}/{}".format(i + 1, len(datas)))
        convert_dataset(ds)


def convert_dataset(ds):
    # float32 air_temperature_pl(time, pressure, ensemble_member, y, x)
    # float32 air_temperature_2m(time, height1, y, x)

    d_data = ds[args.param]
    d_level = ds[args.level]
    d_time = ds["time"]
    d_analysis_time = ds["forecast_reference_time"]
    d_fill_value = ds[args.param]._FillValue

    shp = d_data.shape

    nx = shp[-1]
    ny = shp[-2]

    level_value = int(d_level[0])
    member_value = args.perturbation_number
    at = datetime.datetime.fromtimestamp(int(d_analysis_time[0])).astimezone(pytz.utc)

    for i, vt in enumerate(d_time):
        vt = datetime.datetime.fromtimestamp(int(vt)).astimezone(pytz.utc)

        # level and possible ensemble member dimension value is always
        # zero as each data set is queried separately
        if len(d_data.shape) == 5:
            values = d_data[i, 0, 0, :, :].flatten()
        else:
            values = d_data[i, 0, :, :].flatten()

        convert_to_grib(vt, level_value, member_value, nx, ny, values, d_fill_value)


def convert_to_grib(validtime, level_value, member_value, nx, ny, values, d_fill_value):
    pdtn, dis, cat, num, tosp = param_to_id(args.param)

    year = int(validtime.strftime("%Y"))
    month = int(validtime.strftime("%m"))
    day = int(validtime.strftime("%d"))
    hour = int(validtime.strftime("%H"))

    ft = 0 if pdtn == 1 else -1
    grib = ecc.codes_grib_new_from_samples("GRIB2")
    ecc.codes_set(grib, "tablesVersion", 21)
    ecc.codes_set(grib, "generatingProcessIdentifier", 0)
    ecc.codes_set(grib, "centre", 251)
    ecc.codes_set(grib, "subCentre", 255)
    ecc.codes_set(grib, "gridDefinitionTemplateNumber", 30)
    ecc.codes_set(grib, "latitudeOfSouthernPole", -90000000)
    ecc.codes_set(grib, "longitudeOfSouthernPole", 0)
    ecc.codes_set(grib, "latitudeOfFirstGridPoint", 50319616)
    ecc.codes_set(grib, "longitudeOfFirstGridPoint", 278280)
    ecc.codes_set(grib, "LaD", 63300000)
    ecc.codes_set(grib, "LoV", 15000000)
    ecc.codes_set(grib, "Dx", 2500000)
    ecc.codes_set(grib, "Dy", 2500000)
    ecc.codes_set(grib, "Nx", nx)
    ecc.codes_set(grib, "Ny", ny)
    ecc.codes_set(grib, "Latin1", 63300000)
    ecc.codes_set(grib, "Latin2", 63300000)
    ecc.codes_set(grib, "resolutionAndComponentFlags", 56)
    ecc.codes_set(grib, "discipline", dis)
    ecc.codes_set(grib, "parameterCategory", cat)
    ecc.codes_set(grib, "parameterNumber", num)
    ecc.codes_set(grib, "significanceOfReferenceTime", 1)
    ecc.codes_set(grib, "typeOfProcessedData", 3)
    ecc.codes_set(grib, "productionStatusOfProcessedData", 0)
    ecc.codes_set(grib, "shapeOfTheEarth", 6)
    ecc.codes_set(grib, "productDefinitionTemplateNumber", pdtn)
    ecc.codes_set(grib, "scanningMode", 64)
    ecc.codes_set(grib, "NV", 132)
    ecc.codes_set(grib, "typeOfGeneratingProcess", 4)
    ecc.codes_set(grib, "indicatorOfUnitOfTimeRange", 1)
    ecc.codes_set(grib, "typeOfFirstFixedSurface", level_type_to_id(args.level))
    ecc.codes_set(grib, "level", level_value)
    ecc.codes_set(grib, "numberOfForecastsInEnsemble", 15)
    ecc.codes_set(grib, "perturbationNumber", member_value)
    ecc.codes_set(grib, "year", year)
    ecc.codes_set(grib, "month", month)
    ecc.codes_set(grib, "day", day)
    ecc.codes_set(grib, "hour", hour)
    ecc.codes_set(grib, "forecastTime", ft)
    ecc.codes_set(grib, "bitsPerValue", bits_per_value(args.param))
    ecc.codes_set(grib, "packingType", "grid_ccsds")
    ecc.codes_set(grib, "PVPresent", 1)
    ecc.codes_set_array(grib, "pv", pv())

    if d_fill_value is not None:
        ecc.codes_set(grib, "bitmapPresent", 1)
        ecc.codes_set(grib, "missingValue", d_fill_value)

    ecc.codes_set_values(grib, values)

    if pdtn == 11:
        ecc.codes_set(grib, "lengthOfTimeRange", 1)
        ecc.codes_set(grib, "typeOfStatisticalProcessing", tosp)
        ecc.codes_set(grib, "yearOfEndOfOverallTimeInterval", year)
        ecc.codes_set(grib, "monthOfEndOfOverallTimeInterval", month)
        ecc.codes_set(grib, "dayOfEndOfOverallTimeInterval", day)
        ecc.codes_set(grib, "hourOfEndOfOverallTimeInterval", hour)

    if args.output_filename is None:
        filename = (
            "{}/{}/{:02d}/{:02d}/raw/{}{:02d}{:02d}{:02d}_{}_{}_{}_mbr{}.grib2".format(
                args.output_dir,
                year,
                month,
                day,
                year,
                month,
                day,
                hour,
                common_param_name(args.param),
                common_level_name(args.level),
                level_value,
                member_value,
            )
        )
    else:
        filename = args.output_filename

    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)

    open_mode = "ab" if args.merge else "wb"
    with open(filename, open_mode) as fp:
        ecc.codes_write(grib, fp)


datas = fetch_from_thredds()
convert(datas)
