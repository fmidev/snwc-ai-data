import gridpp
from plotutils import plot
from fileutils import write_grib, read_grib
import numpy as np
import eccodes as ecc
import sys
import pyproj
import requests
import datetime
import argparse
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import fsspec
import os
import time
import copy
import numpy.ma as ma
import warnings
import rioxarray
from flatten_json import flatten
import gzip
from multiprocessing import Process, Queue

warnings.filterwarnings("ignore")

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topography_data", action="store", type=str, required=True)
    parser.add_argument("--landseacover_data", action="store", type=str, required=True)
    parser.add_argument("--parameter", action="store", type=str, required=True)
    parser.add_argument("--parameter_data", action="store", type=str, required=True)
    parser.add_argument("--v_component", action="store", type=str) # if windspeed

    parser.add_argument("--dem_data", action="store", type=str, default="DEM_100m-Int16.tif")
    parser.add_argument("--output", action="store", type=str, required=True)
    parser.add_argument("--output_v", action="store", type=str, required=False) # output for v_component
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--disable_multiprocessing", action="store_true", default=False)

    args = parser.parse_args()

    allowed_params = ["t", "r", "uv", "fg"]
    if args.parameter not in allowed_params:
        print("Error: parameter must be one of: {}".format(allowed_params))
        sys.exit(1)

    return args

def read_netatmo_from_s3(args,fcstime): #tiuha_file):
    # access to tiuha-history bucket
    tiuha_access = os.environ.get("TIUHAS3_ACCESS_KEY")
    tiuha_secret = os.environ.get("TIUHAS3_SECRET_KEY")
    endpoint_url = "https://lake.fmi.fi"
    
    s3_fs = fsspec.filesystem('s3', anon=False, key=tiuha_access, secret=tiuha_secret, client_kwargs={"endpoint_url": endpoint_url})
    qcfile_ls = []
    # create correct bucket name. Bucket with timestamp 2210xx contains obs from 21:50 - 22:00
    time = fcstime
    bucket_time = time.strftime("/%Y/%m/%d/")
    bucket_name = "tiuha-history" + bucket_time
    f2 = time.strftime("%H") + "10"
    # list all files in the S3 bucket
    files = s3_fs.ls(bucket_name) # all files/day
    filtered_files = list(filter(lambda x: f2 in x, files)) # files for hour xx10
    # grep just the ones with correct timestamp
    # data for different countries are in different files
    for file in filtered_files: 
        qcfiles = s3_fs.ls(file)
        filtered_qcfiles = list(filter(lambda x: 'qc_' in x, qcfiles))
        qcfile_ls.append(filtered_qcfiles)

    qcfile_ls = [item for sublist in qcfile_ls for item in sublist]

    #print(qcfile_ls) # correct files for the time to read

    the_obs = pd.DataFrame()  
    for tiuha_file in qcfile_ls:
        uri = "simplecache::{}".format("s3://" + tiuha_file)
        file_obj = fsspec.open_local(
            uri,
            mode="rb",
            s3={"anon": True, "client_kwargs": {"endpoint_url": "https://lake.fmi.fi"}},
        )

        if tiuha_file.endswith('.gz'):
            with gzip.open(file_obj, 'rb') as gz_file:
                utfile = gz_file.read()
        else:
            utfile = file_obj.read()
    
        utfile_str = utfile.decode('utf-8')
        obs = pd.read_json(utfile_str)
        flattened_data = [flatten(feature) for feature in obs["features"]]
        obs = pd.DataFrame(flattened_data)
        obs = obs[obs['properties_observedPropertyTitle'] == 'Air temperature'] # pressure and humidity also available
        obs = obs[obs['properties_qcPassed'] == True]
        obs.drop(obs.columns[[0, 1, 5, 6, 7, 9, 10, 11, 12,14,15,16,17,18,19]], axis=1, inplace=True)
        obs = obs.rename(
            columns={
            "geometry_coordinates_0": "longitude",
            "geometry_coordinates_1": "latitude",
            "geometry_coordinates_2": "station_id",
            "properties_resultTime": "utctime",
            "properties_result": "obs_value",
            }
        )
        # Remove duplicated observations/station by removing duplicated lat/lon values and keep the first value only
        obs = obs[~obs.duplicated(subset=["latitude", "longitude"], keep="first")]
        obs['utctime'] = pd.to_datetime(obs['utctime'])
        #print(obs.head(5))
        obs['utctime'] = obs['utctime'].dt.ceil(freq='H')
        
        # digital elevation map used to interpolate elevation information to all netatmo station points
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
        # reorder/rename columns of the NetAtmo df to match with synop data
        obs = obs[
            [
                "station_id",
                "longitude",
                "latitude",
                "utctime",
                "elevation",
                "obs_value",
            ]
        ]
        # obs.rename(columns={"temperature": "obs_value"}, inplace=True)
        # print(obs.head(10))
        # print("min NetAtmo obs:", min(obs.iloc[:, 5]))
        # print("max NetAtmo obs:", max(obs.iloc[:, 5]))
        the_obs = pd.concat([the_obs, obs], ignore_index=True)
    return the_obs


def read_grid(args):
    """Top function to read "all" gridded data"""

    lons, lats, vals, analysistime, forecasttime = read_grib(args.parameter_data, True)

    _, _, topo, _, _ = read_grib(args.topography_data, False)
    _, _, lc, _, _ = read_grib(args.landseacover_data, False)

    if args.parameter == "uv":
        _, _, v_comp, _, _ = read_grib(args.v_component, False)
        u_comp = vals #vals = np.sqrt(vals ** 2 + wd ** 2)

    # modify  geopotential to height and use just the first grib message, since the topo & lc fields are static
    topo = topo / 9.81
    topo = topo[0]
    lc = lc[0]

    if args.parameter == "t":
        vals = vals - 273.15
    elif args.parameter == "r":
        vals = vals * 100

    grid = gridpp.Grid(lats, lons, topo, lc)
    if args.parameter != "uv": 
        return grid, lons, lats, vals, analysistime, forecasttime, lc, topo
    elif args.parameter == "uv":
        return grid, lons, lats, u_comp, v_comp, analysistime, forecasttime, lc, topo

def read_conventional_obs(args, fcstime, mnwc, analysistime):
    parameter = args.parameter
    # read observations for "analysis time" == leadtime 1
    obstime = fcstime
    # print("Observations are from time:", obstime)

    timestr = obstime.strftime("%Y%m%d%H%M%S")
    trad_obs = []

    # define obs parameter names used in observation database, for ws the potential ws values are used for Finland
    if parameter == "t":
        obs_parameter = "TA_PT1M_AVG"
    elif parameter == "uv":
        obs_parameter = (
            "WSP_PT10M_AVG"  # potential wind speed available for Finnish stations
        )
    elif parameter == "fg":
        obs_parameter = "WG_PT1H_MAX"
    elif parameter == "r":
        obs_parameter = "RH_PT1M_AVG"

    # conventional obs are read from two distinct smartmet server producers
    # if read fails, abort program

    for producer in ["observations_fmi", "foreign"]:
        if producer == "foreign" and parameter == "uv":
            obs_parameter = "WS_PT10M_AVG"
        url = "http://smartmet.fmi.fi/timeseries?producer={}&tz=gmt&precision=auto&starttime={}&endtime={}&param=fmisid,longitude,latitude,utctime,elevation,{}&format=json&keyword=snwc".format(
            producer, timestr, timestr, obs_parameter
        )

        resp = requests.get(url)
        #print(resp.status_code)
        #print(resp.json())

        testitmp = []
        testitmp2 = []
        isresempty = []
        if resp.status_code == 200:
            isresempty = pd.DataFrame(resp.json()).empty # False if resp.json is not empty
            # test if all the retrieved observations are Nan
            if isresempty == False:
                testitmp2 = pd.DataFrame(resp.json())
                testitmp2 = testitmp2[obs_parameter].isnull().all() # True if all obs == NaN
            if testitmp2 == False or isresempty == False: # if all obs are NaN or resp.json=is not empty
                trad_obs += resp.json()       

        if resp.status_code != 200 or testitmp2 == True or isresempty == True:
            print(
                "Not able to connect Smartmet server for observations, original MEPS fields are saved"
            )
            # create obs dataframe with zeros 
            # Since gridding is done for Forecast-Obs difference the zero values wont affect the results
            trad_obs = [{'fmisid': 0, 'longitude': 25, 'latitude': 61, 'utctime': timestr, 'elevation': 0, obs_parameter: 0}]
        #trad_obs += resp.json()

    obs = pd.DataFrame(trad_obs)
    # rename observation column if WS, otherwise WS and WSP won't work
    if parameter == "uv":  # merge columns for WSP and WS
        obs["WSP_PT10M_AVG"] = obs["WSP_PT10M_AVG"].fillna(obs["WS_PT10M_AVG"])

    obs = obs.rename(columns={"fmisid": "station_id"})
    obs = obs.rename(columns={obs.columns[5]: "obs_value"})

    count = len(trad_obs)
    print("Got {} traditional obs stations for time {}".format(count, obstime))

    if count == 0:
        print(
            "Number of observations from Smartmet server is 0, original MNWC fields are saved"
        )

        sys.exit(1)

    # print(obs.head(5))
    print("min obs:", min(obs.iloc[:, 5]))
    print("max obs:", max(obs.iloc[:, 5]))
    return obs


def detect_outliers_zscore(args, fcstime, obs_data):
    # remove outliers based on zscore with separate thresholds for upper and lower tail
    if args.parameter == "r":
        upper_threshold = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        lower_threshold = [-4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5, -4.5]
    elif args.parameter == "t":
        lower_threshold = [-6, -6, -5, -4, -4, -4, -4, -4, -4, -5, -6, -6]
        upper_threshold = [2.5, 2.5, 2.5, 3, 4, 5, 5, 5, 3, 2.5, 2.5, 2.5]
    elif args.parameter == "uv" or args.parameter == "fg":
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

        # for t there's netatmo obs available
        # NetAtmo obs not used!!!
        """
        if args.parameter == "t":
            netatmo = read_netatmo_from_s3(args, fcstime[i])
            print("min NetAtmo obs:", min(netatmo.iloc[:, 5]))
            print("max NetAtmo obs:", max(netatmo.iloc[:, 5]))
            if netatmo is not None:
                obs = pd.concat((obs, netatmo))
        """
        #print("length of all obs:", len(obs))
        outliers, obs = detect_outliers_zscore(args, fcstime[i], obs)
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


def interpolate_single_time(grid, background, points, obs, obs_to_background_variance_ratio, pobs, structure, max_points, idx, q):
    # perform optimal interpolation
    tmp_output = gridpp.optimal_interpolation(
        grid,
        background[idx],
        points[idx],
        obs[idx]["bias"].to_numpy(),
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
    if args.parameter != "uv":
        grid, lons, lats, background, analysistime, forecasttime, lc, topo = read_grid(args)
    elif args.parameter == "uv":
        grid, lons, lats, u_comp, v_comp, analysistime, forecasttime, lc, topo = read_grid(args)
        background = np.sqrt(u_comp ** 2 + v_comp ** 2) # ws
        # calculate wind direction from u and v components
        wd = 180 + np.arctan2(u_comp, v_comp) * 180 / np.pi
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
    # grib file contains 1day worth of data/parameter. MEPS control is run every 3h and 1h and 2h forecasts are used to produce analysis field

    print(analysistime)
    #print(forecasttime)
    # obs is a list of obs for different forecast times
    points, obs = read_obs(args, forecasttime, grid, lc, background, analysistime)
    #print("len points",len(points))
    #print("len fcstime",len(obs))
       
    ot = time.time()
    timedif = ot - et
    print("Reading OBS data takes:", round(timedif, 1), "seconds")
    #print(obs[0])
    
    # Interpolate obs to background grid
    # if obs is just one row (missing) this is still done, but the diff fields are not used
    bias_obs = []
    for i in range(0, len(obs)):
        tmp_obs = obs[i]
        #print(tmp_obs.shape[0])
        # interpolate background to obs points
        tmp_bg_point = gridpp.nearest(grid, points[i], background[i])
        tmp_obs['bias'] = tmp_bg_point - tmp_obs['obs_value']
        # interpolate background0 to obs points
        bias_obs.append(tmp_obs)

    diff = interpolate(grid, points, background0, bias_obs, args)
    intt = time.time()
    timedif = intt - ot
    
    print("Interpolating data takes:", round(timedif, 1), "seconds")
    
    # and convert parameter to T-K or RH-0TO1
    output = []
    output_v = []
    for j in range(0, len(diff)):
        #print(obs[j].shape[0])
        if obs[j].shape[0] == 1: # just one row of obs == missing obs
            tmp_output = background[j]
        else: 
            tmp_output = background[j] - diff[j]
        # Implement simple QC thresholds
        if args.parameter == "r":
            tmp_output = np.clip(tmp_output, 5, 100)  # min RH 5% !
            tmp_output = tmp_output / 100
        elif args.parameter == "uv":
            tmp_output = np.clip(tmp_output, 0, 38)  # max ws same as in oper qc: 38m/s
            # calculate u and v components from ws and wd
            tmp_output_v = tmp_output * np.sin(wd[j] * np.pi / 180) # v-vector
            tmp_output = tmp_output * np.cos(wd[j] * np.pi / 180) # u-vector
        elif args.parameter == "fg":
            tmp_output = np.clip(tmp_output, 0, 50)
        else: # temperature
            tmp_output = tmp_output + 273.15
        
        if args.parameter != "uv":
            output.append(tmp_output)
        elif args.parameter == "uv":    
            output_v.append(tmp_output_v)
            output.append(tmp_output)


    #print("Interpolating forecasts takes:", round(timedif, 1), "seconds")
    # Remove analysistime (leadtime=0), because correction is not made for that time
    #assert len(forecasttime) == len(output)
    #print("output:",output)
    if args.parameter != "uv":
        write_grib(args, args.output, analysistime, forecasttime, output)
    elif args.parameter == "uv":
        write_grib(args, args.output, analysistime, forecasttime, output)
        write_grib(args, args.output_v, analysistime, forecasttime, output_v)

    """
    import matplotlib.pylab as mpl

    # plot diff
    for j in range(0,len(diff)):
        vmin = -5
        vmax = 5
        if args.parameter == "r":
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

if __name__ == "__main__":
    main()
