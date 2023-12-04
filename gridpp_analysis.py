import gridpp
import numpy as np
import eccodes as ecc
import sys
import pyproj
import requests
import datetime
import argparse
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import fsspec
import os
import time
import copy
import numpy.ma as ma
import warnings
import rioxarray
from flatten_json import flatten
from multiprocessing import Process, Queue

warnings.filterwarnings("ignore")


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topography_data", action="store", type=str, required=True)
    parser.add_argument("--landseacover_data", action="store", type=str, required=True)
    parser.add_argument("--parameter", action="store", type=str, required=True)
    parser.add_argument("--parameter_data", action="store", type=str, required=True)
    parser.add_argument(
        "--dem_data", action="store", type=str, default="DEM_100m-Int16.tif"
    )
    parser.add_argument("--output", action="store", type=str, required=True)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--disable_multiprocessing", action="store_true", default=False)

    args = parser.parse_args()

    allowed_params = ["temperature", "humidity", "windspeed", "gust"]
    if args.parameter not in allowed_params:
        print("Error: parameter must be one of: {}".format(allowed_params))
        sys.exit(1)

    return args


def get_shapeofearth(gh):
    """Return correct shape of earth sphere / ellipsoid in proj string format.
    Source data is grib2 definition.
    """

    shape = ecc.codes_get_long(gh, "shapeOfTheEarth")

    if shape == 1:
        v = ecc.codes_get_long(gh, "scaledValueOfRadiusOfSphericalEarth")
        s = ecc.codes_get_long(gh, "scaleFactorOfRadiusOfSphericalEarth")
        return "+R={}".format(v * pow(10, s))

    if shape == 5:
        return "+ellps=WGS84"

    if shape == 6:
        return "+R=6371229.0"

def get_falsings(projstr, lon0, lat0):
    """Get east and north falsing for projected grib data"""

    ll_to_projected = pyproj.Transformer.from_crs("epsg:4326", projstr)
    return ll_to_projected.transform(lat0, lon0)


def get_projstr(gh):
    """Create proj4 type projection string from grib metadata" """

    projstr = None

    proj = ecc.codes_get_string(gh, "gridType")
    first_lat = ecc.codes_get_double(gh, "latitudeOfFirstGridPointInDegrees")
    first_lon = ecc.codes_get_double(gh, "longitudeOfFirstGridPointInDegrees")

    if proj == "polar_stereographic":
        projstr = "+proj=stere +lat_0=90 +lat_ts={} +lon_0={} {} +no_defs".format(
            ecc.codes_get_double(gh, "LaDInDegrees"),
            ecc.codes_get_double(gh, "orientationOfTheGridInDegrees"),
            get_shapeofearth(gh),
        )
        fe, fn = get_falsings(projstr, first_lon, first_lat)
        projstr += " +x_0={} +y_0={}".format(-fe, -fn)

    elif proj == "lambert":
        projstr = (
            "+proj=lcc +lat_0={} +lat_1={} +lat_2={} +lon_0={} {} +no_defs".format(
                ecc.codes_get_double(gh, "Latin1InDegrees"),
                ecc.codes_get_double(gh, "Latin1InDegrees"),
                ecc.codes_get_double(gh, "Latin2InDegrees"),
                ecc.codes_get_double(gh, "LoVInDegrees"),
                get_shapeofearth(gh),
            )
        )
        fe, fn = get_falsings(projstr, first_lon, first_lat)
        projstr += " +x_0={} +y_0={}".format(-fe, -fn)

    else:
        print("Unsupported projection: {}".format(proj))
        sys.exit(1)

    return projstr


def read_file_from_s3(grib_file):
    uri = "simplecache::{}".format(grib_file)

    return fsspec.open_local(
        uri,
        mode="rb",
        s3={"anon": True, "client_kwargs": {"endpoint_url": "https://lake.fmi.fi"}},
    )


def read_grib(gribfile, read_coordinates=False):
    """Read first message from grib file and return content.
    List of coordinates is only returned on request, as it's quite
    slow to generate.
    """
    forecasttime = []
    values = []

    print(f"Reading file {gribfile}")
    wrk_gribfile = gribfile

    if gribfile.startswith("s3://"):
        wrk_gribfile = read_file_from_s3(gribfile)

    lons = []
    lats = []

    with open(wrk_gribfile, "rb") as fp:
        # print("Reading {}".format(gribfile))

        while True:
            try:
                gh = ecc.codes_grib_new_from_file(fp)
            except ecc.WrongLengthError as e:
                print(e)
                file_stats = os.stat(wrk_gribfile)
                print("Size of {}: {}".format(wrk_gribfile, file_stats.st_size))
                sys.exit(1)

            if gh is None:
                break

            ni = ecc.codes_get_long(gh, "Nx")
            nj = ecc.codes_get_long(gh, "Ny")
            dataDate = ecc.codes_get_long(gh, "dataDate")
            dataTime = ecc.codes_get_long(gh, "dataTime")
            forecastTime = ecc.codes_get_long(gh, "endStep")
            analysistime = datetime.datetime.strptime(
                "{}.{:04d}".format(dataDate, dataTime), "%Y%m%d.%H%M"
            )

            ftime = analysistime + datetime.timedelta(hours=forecastTime)
            forecasttime.append(ftime)

            tempvals = ecc.codes_get_values(gh).reshape(nj, ni)
            values.append(tempvals)

            if read_coordinates and len(lons) == 0:
                projstr = get_projstr(gh)

                di = ecc.codes_get_double(gh, "DxInMetres")
                dj = ecc.codes_get_double(gh, "DyInMetres")

                proj_to_ll = pyproj.Transformer.from_crs(projstr, "epsg:4326")

                for j in range(nj):
                    y = j * dj
                    for i in range(ni):
                        x = i * di

                        lat, lon = proj_to_ll.transform(x, y)
                        lons.append(lon)
                        lats.append(lat)

        if read_coordinates == False and len(values) == 1:
            return (
                None,
                None,
                np.asarray(values).reshape(nj, ni),
                analysistime,
                forecasttime,
            )
        elif read_coordinates == False and len(values) > 1:
            return None, None, np.asarray(values), analysistime, forecasttime
        else:
            return (
                np.asarray(lons).reshape(nj, ni),
                np.asarray(lats).reshape(nj, ni),
                np.asarray(values),
                analysistime,
                forecasttime,
            )


def read_grid(args):
    """Top function to read "all" gridded data"""
    # Define the grib-file used as background/"parameter_data"
    #if args.parameter == "temperature":
    #    parameter_data = args.t2_data
    #elif args.parameter == "windspeed":
    #    parameter_data = args.ws_data
    #elif args.parameter == "gust":
    #    parameter_data = args.wg_data
    #elif args.parameter == "humidity":
    #    parameter_data = args.rh_data

    lons, lats, vals, analysistime, forecasttime = read_grib(args.parameter_data, True)

    _, _, topo, _, _ = read_grib(args.topography_data, False)
    _, _, lc, _, _ = read_grib(args.landseacover_data, False)

    # modify  geopotential to height and use just the first grib message, since the topo & lc fields are static
    topo = topo / 9.81
    topo = topo[0]
    lc = lc[0]

    if args.parameter == "temperature":
        vals = vals - 273.15
    elif args.parameter == "humidity":
        vals = vals * 100

    grid = gridpp.Grid(lats, lons, topo, lc)
    return grid, lons, lats, vals, analysistime, forecasttime, lc, topo

def read_conventional_obs(args, fcstime, mnwc, analysistime):
    parameter = args.parameter
    # read observations for "analysis time" == leadtime 1
    obstime = fcstime
    # print("Observations are from time:", obstime)

    timestr = obstime.strftime("%Y%m%d%H%M%S")
    trad_obs = []

    # define obs parameter names used in observation database, for ws the potential ws values are used for Finland
    if parameter == "temperature":
        obs_parameter = "TA_PT1M_AVG"
    elif parameter == "windspeed":
        obs_parameter = (
            "WSP_PT10M_AVG"  # potential wind speed available for Finnish stations
        )
    elif parameter == "gust":
        obs_parameter = "WG_PT1H_MAX"
    elif parameter == "humidity":
        obs_parameter = "RH_PT1M_AVG"

    # conventional obs are read from two distinct smartmet server producers
    # if read fails, abort program

    for producer in ["observations_fmi", "foreign"]:
        if producer == "foreign" and parameter == "windspeed":
            obs_parameter = "WS_PT10M_AVG"
        url = "http://smartmet.fmi.fi/timeseries?producer={}&tz=gmt&precision=auto&starttime={}&endtime={}&param=fmisid,longitude,latitude,utctime,elevation,{}&format=json&keyword=snwc".format(
            producer, timestr, timestr, obs_parameter
        )

        resp = requests.get(url)

        testitmp = []
        testitmp2 = []
        if resp.status_code == 200:
            testitmp2 = pd.DataFrame(resp.json())
            # test if all the retrieved observations are Nan
            testitmp2 = testitmp2[obs_parameter].isnull().all()

        if resp.status_code != 200 or testitmp2 == True or resp.json == testitmp:
            print(
                "Not able to connect Smartmet server for observations, original MNWC fields are saved"
            )
            # Remove analysistime (leadtime=0), because correction is not made for that time
            fcstime.pop(0)
            mnwc = mnwc[1:]
            if parameter == "humidity":
                mnwc = mnwc / 100
            elif parameter == "temperature":
                mnwc = mnwc + 273.15
            write_grib(args, analysistime, fcstime, mnwc)
            sys.exit(1)
        trad_obs += resp.json()

    obs = pd.DataFrame(trad_obs)
    # rename observation column if WS, otherwise WS and WSP won't work
    if parameter == "windspeed":  # merge columns for WSP and WS
        obs["WSP_PT10M_AVG"] = obs["WSP_PT10M_AVG"].fillna(obs["WS_PT10M_AVG"])

    obs = obs.rename(columns={"fmisid": "station_id"})
    obs = obs.rename(columns={obs.columns[5]: "obs_value"})

    count = len(trad_obs)
    print("Got {} traditional obs stations for time {}".format(count, obstime))

    if count == 0:
        print(
            "Number of observations from Smartmet serve is 0, original MNWC fields are saved"
        )
        #fcstime.pop(0)
        #mnwc = mnwc[1:]
        #if parameter == "humidity":
        #    mnwc = mnwc / 100
        #elif parameter == "temperature":
        #    mnwc = mnwc + 273.15
        #write_grib(args, analysistime, fcstime, mnwc)
        sys.exit(1)

    # print(obs.head(5))
    print("min obs:", min(obs.iloc[:, 5]))
    print("max obs:", max(obs.iloc[:, 5]))
    return obs

def read_conventional_obs1(args, fcstime, mnwc, analysistime):
    parameter = args.parameter
    # read observations for "analysis time" == leadtime 1
    #startt = (min(fcstime))
    #endt = (max(fcstime))
    #obstime = fcstime[1]
    # print("Observations are from time:", obstime)

    #timestr1 = startt.strftime("%Y%m%d%H%M%S")
    #timestr2 = endt.strftime("%Y%m%d%H%M%S")

    trad_obs = []
    obs = []

    # define obs parameter names used in observation database, for ws the potential ws values are used for Finland
    if parameter == "temperature":
        obs_parameter = "TA_PT1M_AVG"
    elif parameter == "windspeed":
        obs_parameter = (
            "WSP_PT10M_AVG"  # potential wind speed available for Finnish stations
        )
    elif parameter == "gust":
        obs_parameter = "WG_PT1H_MAX"
    elif parameter == "humidity":
        obs_parameter = "RH_PT1M_AVG"

    # conventional obs are read from two distinct smartmet server producers
    # if read fails, abort program

       # retrieve obs for forecasttime one by one
    for obstime in fcstime:
        timestr = obstime.strftime("%Y%m%d%H%M%S")
        print(obstime)
        for producer in ["observations_fmi", "foreign"]:
            if producer == "foreign" and parameter == "windspeed":
                obs_parameter = "WS_PT10M_AVG"
            url = "http://smartmet.fmi.fi/timeseries?producer={}&tz=gmt&precision=auto&starttime={}&endtime={}&param=fmisid,longitude,latitude,utctime,elevation,{}&format=json&keyword=snwc".format(
                producer, timestr, timestr, obs_parameter
            )

            resp = requests.get(url)
            testitmp = []
            testitmp2 = []
            if resp.status_code == 200:
                testitmp2 = pd.DataFrame(resp.json())
                # test if all the retrieved observations are Nan
                testitmp2 = testitmp2[obs_parameter].isnull().all()

            if resp.status_code != 200 or testitmp2 == True or resp.json == testitmp:
                print(
                    "Not able to connect Smartmet server for observations, original MNWC fields are saved"
                )
                # Remove analysistime (leadtime=0), because correction is not made for that time
                # MUOKKAA tämä, sama logiikka ei toimi
                fcstime.pop(0)
                mnwc = mnwc[1:]
                if parameter == "humidity":
                    mnwc = mnwc / 100
                elif parameter == "temperature":
                    mnwc = mnwc + 273.15
                write_grib(args, analysistime, fcstime, mnwc)
                sys.exit(1)
            trad_obs += resp.json()

        tmp_obs = pd.DataFrame(trad_obs)
        # rename observation column if WS, otherwise WS and WSP won't work
        if parameter == "windspeed":  # merge columns for WSP and WS
            tmp_obs["WSP_PT10M_AVG"] = tmp_obs["WSP_PT10M_AVG"].fillna(tmp_obs["WS_PT10M_AVG"])

        tmp_obs = tmp_obs.rename(columns={"fmisid": "station_id"})

        count = len(trad_obs)
        print("Got {} traditional obs stations for time {}".format(count, obstime))

        if count == 0:
            print(
                "Number of observations from Smartmet server is 0, original MNWC fields are saved"
            )
            #fcstime.pop(0)
            #mnwc = mnwc[1:]
            if parameter == "humidity":
                mnwc = mnwc / 100
            elif parameter == "temperature":
                mnwc = mnwc + 273.15
            write_grib(args, analysistime, fcstime, mnwc)
            sys.exit(1)

        # print(obs.head(5))
        print("min obs:", min(tmp_obs.iloc[:, 5]))
        print("max obs:", max(tmp_obs.iloc[:, 5]))
        obs.append(tmp_obs)
    
    print(len(obs))
    return obs


def read_netatmo_obs(args, fcstime):
    # read Tiuha db NetAtmo observations for "analysis time" == leadtime 1
    snwc1_key = os.environ.get("SNWC1_KEY")
    assert snwc1_key is not None, "tiuha api key not find (env variable 'SNWC1_KEY')"

    os.environ["NO_PROXY"] = "tiuha-dev.apps.ock.fmi.fi"
    obstime = fcstime[1]

    url = "https://tiuha-dev.apps.ock.fmi.fi/v1/edr/collections/netatmo-air_temperature/cube?bbox=4,54,32,71.5&start={}Z&end={}Z".format(
        (obstime - datetime.timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%S"),
        obstime.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    headers = {"Authorization": f"Basic {snwc1_key}"}
    # os.environ['NO_PROXY'] = 'tiuha-dev.apps.ock.fmi.fi'
    resp = requests.get(url, headers=headers)

    crowd_obs = None
    testitmp = []
    testitmp2 = []

    if resp.status_code == 200:
        testitmp2 = pd.DataFrame(resp.json())

    if resp.status_code != 200 or resp.json() == testitmp or len(testitmp2) == 0:
        print("Error fetching NetAtmo data, status code: {}".format(resp.status_code))
    else:
        crowd_obs = resp.json()
        # print("Got {} crowd sourced obs stations".format(len(crowd_obs)))

    obs = None

    if crowd_obs is not None:
        flattened_data = [flatten(feature) for feature in crowd_obs["features"]]
        obs = pd.DataFrame(flattened_data)
        obs.drop(obs.columns[[0, 1, 5, 6, 7, 9, 10, 11, 12]], axis=1, inplace=True)
        obs = obs.rename(
            columns={
                "geometry_coordinates_0": "longitude",
                "geometry_coordinates_1": "latitude",
                "geometry_coordinates_2": "station_id",
                "properties_resultTime": "utctime",
                "properties_result": "temperature",
            }
        )
        # Remove duplicated observations/station by removing duplicated lat/lon values and keep the first value only
        obs = obs[~obs.duplicated(subset=["latitude", "longitude"], keep="first")]
        print("Got {} crowd sourced obs stations".format(len(obs)))

        # netatmo obs do not contain elevation information, but we need thatn
        # to have the best possible result from optimal interpolation
        #
        # use digital elevation map data to interpolate elevation information
        # to all netatmo station points

        # print("Interpolating elevation to NetAtmo stations")
        dem = rioxarray.open_rasterio(args.dem_data)

        # dem is projected to lambert, our obs data is in latlon
        # transform latlons to projected coordinates

        ll_to_proj = pyproj.Transformer.from_crs("epsg:4326", dem.rio.crs)
        xs, ys = ll_to_proj.transform(obs["latitude"], obs["longitude"])
        obs["x"] = xs
        obs["y"] = ys

        # interpolated dem data to netatmo station points in x,y coordinates

        demds = dem.to_dataset("band").rename({1: "dem"})
        x = demds["x"].values

        # RegularGridInterpolator requires y axis value to be ascending -
        # geotiff is always descending

        y = np.flip(demds["y"].values)
        z = np.flipud(demds["dem"].values)

        interp = RegularGridInterpolator(points=(y, x), values=z)

        points = np.column_stack((obs["y"], obs["x"]))
        obs["elevation"] = interp(points)

        obs = obs.drop(columns=["x", "y"])
        # print(obs.head(5))
        # reorder/rename columns of the NetAtmo df to match with synop data
        obs = obs[
            [
                "station_id",
                "longitude",
                "latitude",
                "utctime",
                "elevation",
                "temperature",
            ]
        ]
        obs.rename(columns={"temperature": "TA_PT1M_AVG"}, inplace=True)
        # print(obs.head(10))
        print("min NetAtmo obs:", min(obs.iloc[:, 5]))
        print("max NetAtmo obs:", max(obs.iloc[:, 5]))

    return obs


def detect_outliers_zscore(args, fcstime, obs_data):
    # remove outliers based on zscore with separate thresholds for upper and lower tail
    if args.parameter == "humidity":
        upper_threshold = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        lower_threshold = [-4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5]
    elif args.parameter == "temperature":
        lower_threshold = [-6, -6, -5, -4, -4, -4, -4, -4, -4, -5, -6, -6]
        upper_threshold = [2.5, 2.5, 2.5, 3, 4, 5, 5, 5, 3, 2.5, 2.5, 2.5]
    elif args.parameter == "windspeed" or args.parameter == "gust":
        upper_threshold = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
        lower_threshold = [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4]

    thres_month = fcstime.month
    up_thres = upper_threshold[thres_month - 1]
    low_thres = lower_threshold[thres_month - 1]

    outliers = []

    tmpobs = obs_data.iloc[:, 5]
    mean = np.mean(tmpobs)
    std = np.std(tmpobs)
    for i in tmpobs:
        z = (i - mean) / std
        if z > up_thres or z < low_thres:
            outliers.append(i)
    dataout = obs_data[~obs_data.iloc[:, 5].isin(outliers)]
    # print(obs_data[obs_data.iloc[:,5].isin(outliers)])
    return outliers, dataout

def read_obs(args, fcstime, grid, lc, mnwc, analysistime):
    """Read observations from smartmet server"""
    obsis = []
    points = []
    for i in range(0, len(fcstime)):
        obs = read_conventional_obs(args, fcstime[i], mnwc, analysistime)

    # for temperature there's netatmo obs available
        if args.parameter == "temperature":
            netatmo = read_netatmo_obs(args, fcstime)
            if netatmo is not None:
                obs = pd.concat((obs, netatmo))

            # obs["temperature"] += 273.15

        outliers, obs = detect_outliers_zscore(args, fcstime[1], obs)
        print("removed " + str(len(outliers)) + " outliers from observations")
        # print(outliers)

        print("min of QC obs:", min(obs.iloc[:, 5]))
        print("max of QC obs:", max(obs.iloc[:, 5]))

        points1 = gridpp.Points(
            obs["latitude"].to_numpy(),
            obs["longitude"].to_numpy(),
        )
        # interpolate nearest land sea mask values from grid to obs points (NWP data used, since there's no lsm info from obs stations available)
        obs["lsm"] = gridpp.nearest(grid, points1, lc)

        tmp_points = gridpp.Points(
        obs["latitude"].to_numpy(),
        obs["longitude"].to_numpy(),
        obs["elevation"].to_numpy(),
        obs["lsm"].to_numpy(),
        )
        obsis.append(obs)
        points.append(tmp_points)
    return points, obsis


def write_grib_message(fp, args, analysistime, forecasttime, data):
    pdtn = 70
    tosp = None
    if args.parameter == "humidity":
        levelvalue = 2
        pcat = 1
        pnum = 192
    elif args.parameter == "temperature":
        levelvalue = 2
        pnum = 0
        pcat = 0
    elif args.parameter == "windspeed":
        pcat = 2
        pnum = 1
        levelvalue = 10
    elif args.parameter == "gust":
        levelvalue = 10
        pnum = 22
        pcat = 2
        pdtn = 72
        tosp = 2
    # Store different time steps as grib msgs
    for j in range(0, len(data)):
        tdata = data[j]
        forecastTime = int((forecasttime[j] - analysistime).total_seconds() / 3600)
        print(forecasttime[j])
        print(forecastTime)

        # - For non-aggregated parameters, grib2 key 'forecastTime' is the time of the forecast
        # - For aggregated parameters, it is the start time of the aggregation period. The end of
        #   the period is defined by 'lengthOfTimeRange'
        #   Because snwc is in hourly time steps, reduce forecast time by one

        if tosp == 2:
            forecastTime -= 1
        print(tosp)
        #assert (tosp is None and j + 1 == forecastTime) or (
        #    tosp == 2 and j == forecastTime
        #)
        h = ecc.codes_grib_new_from_samples("regular_ll_sfc_grib2")
        ecc.codes_set(h, "tablesVersion", 28)
        ecc.codes_set(h, "gridType", "lambert")
        ecc.codes_set(h, "shapeOfTheEarth", 5)
        ecc.codes_set(h, "Nx", tdata.shape[1])
        ecc.codes_set(h, "Ny", tdata.shape[0])
        ecc.codes_set(h, "DxInMetres", 2370000 / (tdata.shape[1] - 1))
        ecc.codes_set(h, "DyInMetres", 2670000 / (tdata.shape[0] - 1))
        ecc.codes_set(h, "jScansPositively", 1)
        ecc.codes_set(h, "latitudeOfFirstGridPointInDegrees", 50.319616)
        ecc.codes_set(h, "longitudeOfFirstGridPointInDegrees", 0.27828)
        ecc.codes_set(h, "Latin1InDegrees", 63.3)
        ecc.codes_set(h, "Latin2InDegrees", 63.3)
        ecc.codes_set(h, "LoVInDegrees", 15)
        ecc.codes_set(h, "LaDInDegrees", 63.3)
        ecc.codes_set(h, "latitudeOfSouthernPoleInDegrees", -90)
        ecc.codes_set(h, "longitudeOfSouthernPoleInDegrees", 0)
        ecc.codes_set(h, "dataDate", int(analysistime.strftime("%Y%m%d")))
        ecc.codes_set(h, "dataTime", int(analysistime.strftime("%H%M")))
        ecc.codes_set(h, "forecastTime", forecastTime)
        ecc.codes_set(h, "centre", 86)
        ecc.codes_set(h, "generatingProcessIdentifier", 203)
        ecc.codes_set(h, "discipline", 0)
        ecc.codes_set(h, "parameterCategory", pcat)
        ecc.codes_set(h, "parameterNumber", pnum)
        ecc.codes_set(h, "productDefinitionTemplateNumber", pdtn)
        if tosp is not None:
            ecc.codes_set(h, "typeOfStatisticalProcessing", tosp)
            ecc.codes_set(h, "lengthOfTimeRange", 1)
            ecc.codes_set(
                h, "yearOfEndOfOverallTimeInterval", int(forecasttime[j].strftime("%Y"))
            )
            ecc.codes_set(
                h,
                "monthOfEndOfOverallTimeInterval",
                int(forecasttime[j].strftime("%m")),
            )
            ecc.codes_set(
                h, "dayOfEndOfOverallTimeInterval", int(forecasttime[j].strftime("%d"))
            )
            ecc.codes_set(
                h, "hourOfEndOfOverallTimeInterval", int(forecasttime[j].strftime("%H"))
            )
            ecc.codes_set(h, "minuteOfEndOfOverallTimeInterval", 0)
            ecc.codes_set(h, "secondOfEndOfOverallTimeInterval", 0)
        ecc.codes_set(h, "typeOfFirstFixedSurface", 103)
        ecc.codes_set(h, "scaledValueOfFirstFixedSurface", levelvalue)
        ecc.codes_set(h, "packingType", "grid_ccsds")
        ecc.codes_set(h, "indicatorOfUnitOfTimeRange", 1)  # hours
        ecc.codes_set(h, "typeOfGeneratingProcess", 2)  # deterministic forecast
        ecc.codes_set(h, "typeOfProcessedData", 2)  # analysis and forecast products
        ecc.codes_set_values(h, tdata.flatten())
        ecc.codes_write(h, fp)
    ecc.codes_release(h)


def write_grib(args, analysistime, forecasttime, data):
    if args.output.startswith("s3://"):
        openfile = fsspec.open(
            "simplecache::{}".format(args.output),
            "wb",
            s3={
                "anon": False,
                "key": os.environ["S3_ACCESS_KEY_ID"],
                "secret": os.environ["S3_SECRET_ACCESS_KEY"],
                "client_kwargs": {"endpoint_url": "https://lake.fmi.fi"},
            },
        )
        with openfile as fpout:
            write_grib_message(fpout, args, analysistime, forecasttime, data)
    else:
        with open(args.output, "wb") as fpout:
            write_grib_message(fpout, args, analysistime, forecasttime, data)

    print(f"Wrote file {args.output}")


def interpolate_single_time(grid, background, points, obs, obs_to_background_variance_ratio, pobs, structure, max_points, idx, q):
    # perform optimal interpolation
    tmp_output = gridpp.optimal_interpolation(
        grid,
        background[idx],
        points[idx],
        obs[idx]["obs_value"].to_numpy(),
        obs_to_background_variance_ratio[idx],
        pobs[idx],
        structure,
        max_points,
    )

    print(
        "step {} min grid: {:.1f} max grid: {:.1f}".format(
            idx, np.amin(tmp_output), np.amax(tmp_output)
        )
    )

    if q is not None:
        # return index and output, so that the results can
        # later be sorted correctly
        q.put((idx, tmp_output))
    else:
        return tmp_output


def interpolate(grid, points, background, obs, args):
    # interpolate(grid, obs, background, args, lc)
    """Perform optimal interpolation"""

    output = []
    # create a mask to restrict the modifications only to land area (where lc = 1)
    #lc0 = np.logical_not(lc).astype(int)

    # Interpolate background data to observation points
    pobs = []
    obs_to_background_variance_ratio = []
    for i in range(0, len(obs)):
        # interpolate background to obs points
        pobs.append(gridpp.nearest(grid, points[i], background[i]))
        obs_to_background_variance_ratio.append(np.full(points[i].size(), 0.1))
        
    # Barnes structure function with horizontal decorrelation length 30km, vertical decorrelation length 200m
    structure = gridpp.BarnesStructure(30000, 200, 0.5)

    # Include at most this many "observation points" when interpolating to a grid point
    max_points = 20

    # error variance ratio between observations and background
    # smaller values -> more trust to observations
    #obs_to_background_variance_ratio = np.full(points.size(), 0.1)

    if args.disable_multiprocessing:
        output = [
            interpolate_single_time(
                grid,
                background,
                points,
                obs,
                obs_to_background_variance_ratio,
                pobs,
                structure,
                max_points,
                x,
                None,
            )
            for x in range(len(obs))
        ]

    else:
        q = Queue()
        processes = []
        outputd = {}

        for i in range(len(obs)):
            processes.append(
                Process(
                    target=interpolate_single_time,
                    args=(
                        grid,
                        background,
                        points,
                        obs,
                        obs_to_background_variance_ratio,
                        pobs,
                        structure,
                        max_points,
                        i,
                        q,
                    ),
                )
            )
            processes[-1].start()

        for p in processes:
            # get return values from queue
            # they might be in any order (non-consecutive)
            ret = q.get()
            outputd[ret[0]] = ret[1]

        for p in processes:
            p.join()

        for i in range(len(obs)):
            # sort return values from 0 to 8
            output.append(outputd[i])

    return output


def main():
    args = parse_command_line()

    # print("Reading NWP data for", args.parameter )
    # read in the parameter which is forecasted
    # background contains mnwc values for different leadtimes
    st = time.time()
    grid, lons, lats, background, analysistime, forecasttime, lc, topo = read_grid(args)
    # create "zero" background for interpolating the bias
    background0 = copy.copy(background)
    background0[background0 != 0] = 0

    #ws, rh, t2, wg, cl, ps, wd, q2 = read_ml_grid(args)
    et = time.time()
    timedif = et - st
    print(
        "Reading NWP data for", args.parameter, "takes:", round(timedif, 1), "seconds"
    )

    # Read observations from smartmet server
    # Use correct time!!!
    # grib file contains 1day worth if data/parameter. MEPS control is run every 3h and 1h and 2h forecasts are used to produce analysis field

    print(analysistime)
    print(forecasttime)
    # obs is a list of obs for different forecast times
    points, obs = read_obs(args, forecasttime, grid, lc, background, analysistime)
    print("len points",len(points))
    print("len fcstime",len(obs))

    ot = time.time()
    timedif = ot - et
    print("Reading OBS data takes:", round(timedif, 1), "seconds")
    print(obs[0])
    
    # Interpolate ML point forecasts for bias correction + 0h analysis time
    output = interpolate(grid, points, background, obs, args)
    print(len(output))
    print(len(forecasttime))
    
    #print("Interpolating forecasts takes:", round(timedif, 1), "seconds")
    
    # Remove analysistime (leadtime=0), because correction is not made for that time
    assert len(forecasttime) == len(output)
    write_grib(args, analysistime, forecasttime, output)

    """
    import matplotlib.pylab as mpl

    # plot diff
    for j in range(0,len(diff)):
        vmin = -5
        vmax = 5
        if args.parameter == "humidity":
             vmin, vmax = -50, 50
        mpl.pcolormesh(lons, lats, diff[j], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        mpl.xlim(0, 35)
        mpl.ylim(55, 75)
        mpl.gca().set_aspect(2)
        mpl.savefig('diff' + args.parameter + str(j) + '.png')
        #mpl.show()
    for k in range(0,len(output)):
        vmin = np.min(output[k])
        vmax = np.max(output[k])
        mpl.pcolormesh(lons, lats, output[k], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        mpl.xlim(0, 35)
        mpl.ylim(55, 75)
        mpl.gca().set_aspect(2)
        mpl.savefig('output' + args.parameter + str(k) + '.png')
        #mpl.show()
    """
    if args.plot:
        plot(obs, background, output, diff, lons, lats, args)


def plot(obs, background, output, diff, lons, lats, args):
    import matplotlib.pyplot as plt

    vmin1 = -5
    vmax1 = 5
    if args.parameter == "temperature":
        obs_parameter = "TA_PT1M_AVG"
        output = list(map(lambda x: x - 273.15, output))
    elif args.parameter == "windspeed":
        obs_parameter = "WSP_PT10M_AVG"
    elif args.parameter == "gust":
        obs_parameter = "WG_PT1H_MAX"
    elif args.parameter == "humidity":
        obs_parameter = "RH_PT1M_AVG"
        output = np.multiply(output, 100)
        vmin1 = -30
        vmax1 = 30

    vmin = min(np.amin(background), np.amin(output))
    vmax = min(np.amax(background), np.amax(output))

    # vmin1 =  np.amin(diff)
    # vmax1 =  np.amax(diff)

    for k in range(0, len(diff)):
        plt.figure(figsize=(13, 6), dpi=80)

        plt.subplot(1, 3, 1)
        plt.pcolormesh(
            np.asarray(lons),
            np.asarray(lats),
            background[k + 1],
            cmap="Spectral_r",  # "RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )

        plt.xlim(0, 35)
        plt.ylim(55, 75)
        cbar = plt.colorbar(
            label="MNWC " + str(k) + "h " + args.parameter, orientation="horizontal"
        )

        plt.subplot(1, 3, 2)
        plt.pcolormesh(
            np.asarray(lons),
            np.asarray(lats),
            diff[k],
            cmap="RdBu_r",
            vmin=vmin1,
            vmax=vmax1,
        )

        """
        plt.scatter(
        obs["longitude"],
        obs["latitude"],
        s=10,
        c=obs[obs_parameter],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        )
        """
        plt.xlim(0, 35)
        plt.ylim(55, 75)
        cbar = plt.colorbar(
            label="Diff " + str(k) + "h " + args.parameter, orientation="horizontal"
        )

        plt.subplot(1, 3, 3)
        plt.pcolormesh(
            np.asarray(lons),
            np.asarray(lats),
            output[k],
            cmap="Spectral_r",
            vmin=vmin,
            vmax=vmax,
        )

        plt.xlim(0, 35)
        plt.ylim(55, 75)
        cbar = plt.colorbar(
            label="XGB " + str(k) + "h " + args.parameter, orientation="horizontal"
        )

        # plt.show()
        plt.savefig("all_" + args.parameter + str(k) + ".png")


if __name__ == "__main__":
    main()
