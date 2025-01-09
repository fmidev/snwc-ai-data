from fileutils import write_grib, read_grib
import numpy as np
import eccodes as ecc
import argparse
import pandas as pd
from scipy.ndimage import uniform_filter
import xarray as xr
import datetime 
from datetime import timedelta
import os
from cc_plotting import plot_single_field, plot_single_border, plot_two_border
import tempfile
import gc
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#from mpl_toolkits.basemap import Basemap

#warnings.filterwarnings("ignore")
tempfile.tempdir = "/home/users/hietal/data/tmp"

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", action="store", type=str, required=True)
    parser.add_argument("--end", action="store", type=str, required=True)
    parser.add_argument("--output", action="store", type=str, required=True)

    args = parser.parse_args()
    return args

def read_grid(args):
    """Top function to read "all" gridded data"""

    lons, lats, vals, analysistime, forecasttime = read_grib(args.rh, True)
    _, _, nl, _, _ = read_grib(args.nl, False) 
    _, _, nm, _, _ = read_grib(args.nm, False)
    _, _, nh, _, _ = read_grib(args.nh, False)

    return lons, lats, vals, analysistime, forecasttime, nl, nm, nh

def read_zarr(filen):
    storage_options = {
        "anon": True,
        "client_kwargs": {"endpoint_url": "https://lake.fmi.fi"},
    }
    ds = xr.open_zarr(filen, storage_options=storage_options)
    return ds

def read_arcus_daily(year, month, dd, timedate, param):
    HH = ["00", "03", "06", "09", "12", "15", "18", "21"]
    s3_arcus = "s3://calibration/MEPS_prod/{year}/{month}/{dd}/{hh}/mbr{member:03d}/{ctype}_heightAboveGround_0_instant_{timedate}{hh}_mbr{member:03d}.grib2"
    #s3://calibration/MEPS_prod/2023/11/23/12/mbr000/hcc_heightAboveGround_0_instant_2023112312_mbr000.grib2
    vals = []
    for hh in HH:
        member = 0
        s3_path = s3_arcus.format(year=year, month=month, dd=dd, hh=hh, member=member, ctype=param, timedate=timedate)
        try:
            _, _, val, _, _ = read_grib(s3_path, False, "arcus")
        except FileNotFoundError:
            member = 1
            s3_path = s3_arcus.format(year=year, month=month, dd=dd, hh=hh, member=member, ctype=param, timedate=timedate)
            try:
                _, _, val, _, _ = read_grib(s3_path, False, "arcus")
            except FileNotFoundError:
                #print(f"No data found for HH {hh}, mbr00 or mbr01. Skipping.")
                continue
        # select only 3 first hours i.e. arrays
        val = val[0:3]
        vals.append(val)
    vals = np.concatenate(vals, axis=0)
    return vals

def dummy_nwcsaf(x1,x2, missing=False):
    shape = (24, 1069, 949)
    missing_fraction = 0.1  
    random_array = np.random.uniform(x1, x2, size=shape)
    random_array = np.round(random_array).astype(int)
    if missing:
        missing_mask = np.random.rand(*shape) < missing_fraction
        random_array = random_array.astype(float)
        random_array[missing_mask] = np.nan
    return random_array

def main():
    args = parse_command_line()
    start = datetime.datetime.strptime(args.start, "%Y-%m-%d")
    sY = start.strftime("%Y")
    sM = start.strftime("%m")
    sD = start.strftime("%d")
    end = datetime.datetime.strptime(args.end, "%Y-%m-%d")
    date_series = pd.date_range(start=args.start, end=args.end, freq='D')
    # read srad data
    sr = read_zarr("s3://ecmwf-ssrd-data/ssrd.zarr")
    #if not os.path.exists(args.output):
    ## Initialize Zarr store once
    #template.to_zarr(args.output, mode="w", consolidated=True)
    template = sr.copy(deep=True) # copy template for output data
    template = template.drop_vars("ssrd").assign_coords(time=[])
    template = template.load()
    #template.to_zarr(args.output, mode="w") 
    y = template["y"].values
    x = template["x"].values
    # read clearsky data
    cs = xr.open_zarr("/home/users/hietal/projects/snwc-ai-data/eff_cloudiness_pp/clearsky.zarr")
    # read nwcsaf data (effc, cttp), might have missing times!
    nwcsaf = read_zarr("s3://nwcsaf-archive-data/nwcsaf.zarr")
    #print("DS output",nwcsaf)
    # s3 template
    s3_rh = "s3://meps-ai-data/meps/{year}/{month}/{dd}/{timedate}_rcorr_heightAboveGround_2.grib2"
    
    cloud = []
    for day in date_series:
        print(day)
        year = day.strftime("%Y")
        month = day.strftime("%m")
        dd = day.strftime("%d")
        timedate = f"{year}{month}{dd}"
        xa_day = day.strftime("%Y-%m-%d")
        # Replace the year for clearsky data (for 2023 months 4...9 only!)
        date23 = day.replace(year=2023)
        cs_day = date23.strftime("%Y-%m-%d")
        # read meps rh (the whole day in one grib file)
        s3_path = s3_rh.format(year=year, month=month, dd=dd, timedate=timedate)
        lons, lats, rh, analysistime, forecasttime = read_grib(s3_path, True)
        ## create dummy data for effc, cttp 
        #effc = dummy_nwcsaf(0, 100)
        #cttp = dummy_nwcsaf(-80, 0, True)
        # read arcus data (nl, nm, nh) from control member 0-2h forecasts aggregated as daily values
        nh = read_arcus_daily(year, month, dd, timedate, "hcc")
        nm = read_arcus_daily(year, month, dd, timedate, "mcc")
        nl = read_arcus_daily(year, month, dd, timedate, "lcc")
        for hour in range(0, 24):
            # create time for hourly data
            timehour = day + timedelta(hours=hour)
            try: # check if nwcsaf is missing for the time
                nwcsaf1 = nwcsaf.sel(time=timehour)
                nwcsaf1 = nwcsaf.sel(time=timehour)
                effc_tmp = nwcsaf1["effective_cloudiness"].values * 100
                # replace na's in effc_tmp with 0
                effc_tmp[np.isnan(effc_tmp)] = 0
                # copy effc_tmp field for plotting
                #effc_copy = effc_tmp.copy()
                cttp_tmp = nwcsaf1["cloudtop_temperature"].values - 273.15
                #plot_single_border(effc_tmp, lats, lons, "effc", timehour, 0, 100, 60, 70, 20, 33) 
                #plot_single_field(cttp_tmp, "cttp", timehour, template, -80, 5)
                nh_tmp = nh[hour] * 100
                nm_tmp = nm[hour] * 100
                nl_tmp = nl[hour] * 100
                rh_tmp = rh[hour] * 100
                avg1 = uniform_filter(effc_tmp, size=3, mode='reflect')#
                #plot_single_field(avg1, "avg1", timehour, template, 0, 100)
                avg2 = uniform_filter(effc_tmp, size=7, mode='reflect') #
                #plot_single_field(avg2, "avg2", timehour, template, 0, 100)
                if (day.month >= 10 or day.month <= 3): # correction for wintertime 
                    # create cloudmask based on cttp
                    effc_tmp[(np.isfinite(cttp_tmp)) & (effc_tmp == 0)] = 80
                    # average out the small gaps in data
                    effc_tmp[(effc_tmp <= 90) & (avg1 > 65)] = 90
                    effc_tmp[(effc_tmp <= 90) & (avg2 > 55)] = 90
                    # reduce cloud cover if just high level clouds
                    # if (effc <10  && rh >= 86 and (nl > 80 or cmqc == 24) {effc = 60 }
                    effc_tmp[(effc_tmp < 10) & (rh_tmp >= 86) & (nl_tmp > 80)] = 60
                    # if (effc <=60  && rh >= 98 and (nl > 20 or cmqc == 24 ) {pilvi = 80 }  
                    effc_tmp[(effc_tmp <= 60) & (rh_tmp >= 98) & (nl_tmp > 20)] = 80
                elif (day.month >= 4 or day.month <= 9): # summer correction
                    # radiation parameters: short wave and clear sky 
                    sr1 = sr.sel(time=slice(xa_day, xa_day))
                    cs1 = cs.sel(time=slice(cs_day, cs_day))
                    sr_tmp1 = sr1["ssrd"].values
                    cs_tmp1 = cs1["clearsky"].values
                    sr_tmp = sr_tmp1[hour]
                    cs_tmp = cs_tmp1[hour]
                    # reduce if just high level clouds
                    effc_tmp[(effc_tmp > 50) & (nh_tmp > 50) & (nl_tmp < 20) & (nm_tmp < 20)] = 50
                    # average out the small gaps in data
                    effc_tmp[(effc_tmp <= 80) & (avg2 > 60)] = 60
                    effc_tmp[(effc_tmp <= 70) & (avg1 > 50)] = 70
                    # reduce cloud cover if cumulus in midday
                    if (hour >= 7 and hour < 17): 
                        sr_ratio = np.full_like(sr_tmp, np.nan)
                        # avoid divide by zero and only apply if srad is above 100 W/m2
                        valid_mask = sr_tmp >= 100
                        sr_ratio[valid_mask] = sr_tmp[valid_mask] / cs_tmp[valid_mask]
                        effc_tmp[(sr_ratio >= 0.65) & (effc_tmp > 80)] = 80
                        effc_tmp[(sr_ratio >= 0.75) & (effc_tmp > 50)] = 50
                #plot_single_border(effc_tmp, lats, lons, "effc_pp", timehour, 0, 100, 60, 70, 20, 33)        
                #plot_single_field(effc_tmp, "effc_pp", timehour, template, 0, 100)   
                ###plot_two_border(effc_copy, effc_tmp, lats, lons, "effc", "effc_pp", timehour, 0, 100, 60, 70, 20, 33)
                # data start time "reference time"
                ref_time = datetime.datetime(int(sY), int(sM), int(sD), 0, 0, 0)
                # time of the hourly data
                timedd = datetime.datetime(int(year), int(month), int(dd), hour, 0, 0)
                hourly_ds = xr.Dataset(
                    {
                        "tcc": (("time", "y", "x"), effc_tmp[np.newaxis, ...]),
                    },
                    coords={
                        "time": [timedd],
                        "y": y,
                        "x": x,
                    },
            
                )
                hourly_ds["time"].attrs = {}  # Clear any residual attributes
                hourly_ds["time"].encoding = {
                "units": f"hours since {ref_time.isoformat()}",
                "calendar": "proleptic_gregorian",
                "dtype": "int64",  
                }
                chunk_sizes = {"time": 1, "y": 1069, "x": 949}
                hourly_ds = hourly_ds.chunk(chunk_sizes)
                hourly_ds["spatial_ref"] = xr.DataArray(
                    data=np.array(0),  # needs dummy data
                    attrs=nwcsaf["spatial_ref"].attrs  # Copy attributes from nwscf
                )
                # Add the grid_mapping attribute to variable
                hourly_ds["tcc"].attrs["grid_mapping"] = "spatial_ref"

                # Check if Zarr file exists
                if not os.path.exists(args.output):
                    # First time: Create the Zarr store
                    hourly_ds.to_zarr(args.output, mode="w", consolidated=True)
                    #print(f"Initialized Zarr store with time: {timedd}")
                else:
                    # Subsequent times: Append to the Zarr store
                    hourly_ds.to_zarr(args.output, mode="a", append_dim="time")
                    #print(f"Appended time: {timedd}")
                    gc.collect()
                
                
            except KeyError:
                #print("No data for time:", timehour)
                continue
    
if __name__ == "__main__":
    main()