# MEPS data archive for AI model training

A timeseries of MEPS data has been downloaded to FMI s3 storage to enable easier development of AI models. The data is in grib2 format for now, alternative formats are considered.

This is an analysis data set: for each time in the time series we store the analysis field or the shortest forecast available. There are no actual forecasts in the archive. This means that for example for two meter temperature for one day we have 24 fields.

## Data fetching

Data is fetched from two archives: Metcoop archive arcus, and MET.no archive threadds. Neither archive provides all MEPS parameters, but in general arcus has the data in better time resolution, but MET.no archive has more parameters available.

We have fetched primarily from arcus, and used thredds when arcus does not have a certain parameter stored.

All data is stored as grib2. Thredds data is equivalent to arcus data except for mean sea level pressure, where it seems that some sort of post processing has been done.

Currently (2023-12-11) s3 bucket has data for 2021-04 to 2023-02, but it is being updated. The aim is to provide data from 2021 to current date.

## Parameters

The following parameters from MEPS dmo are stored in this archive:

* wind gust speed (level: 10m above ground)
  * grib name: fg
* mixed layer depth (0m above ground)
  * grib name: mld
* pressure (mean sea level, 0m above ground)
  * grib name: pres
* relative humidity (2m above ground, pressure levels 300,500,700,850,925,1000)
  * grib name: r
* temperature (0m above ground, 2m above ground, pressure levels 300,500,700,850,925,1000)
  * grib name: t
* u wind (10m above ground, pressure levels 300,500,700,850,925,1000)
  * grib name: u
* v wind (10m above ground, pressure levels 300,500,700,850,925,1000)
  * grib name: v
* geopotential (pressure levels 300,500,700,850,925,1000)
  * grib name: z

The following observation-corrected parameters are also stored:

* wind gust speed (level: 10m above ground)
  * grib name: fgcor
* relative humidity (2m above ground)
  * grib name: rcor
* temperature (2m above ground)
  * grib name: tcor
* u wind (10m above ground)
  * grib name: ucor
* v wind (10m above ground)
  * grib name: vcor


## How to access data

Accessible only inside FMI networks.

Data is in daily "directories" in s3.

```
$ s3cmd ls s3://meps-ai-data/meps/2021/04/01/
2023-12-04 14:09     33196589  s3://meps-ai-data/meps/2021/04/01/20210401_fg_heightAboveGround_10.grib2
2023-12-04 14:09     31316003  s3://meps-ai-data/meps/2021/04/01/20210401_mld_heightAboveGround_0.grib2
2023-12-04 14:09     25027757  s3://meps-ai-data/meps/2021/04/01/20210401_pres_heightAboveSea_0.grib2
2023-12-04 14:09     33045610  s3://meps-ai-data/meps/2021/04/01/20210401_r_heightAboveGround_2.grib2
2023-12-04 14:09     32190272  s3://meps-ai-data/meps/2021/04/01/20210401_r_isobaricInhPa_1000.grib2
2023-12-04 14:09     29071039  s3://meps-ai-data/meps/2021/04/01/20210401_r_isobaricInhPa_300.grib2
2023-12-04 14:09     29030401  s3://meps-ai-data/meps/2021/04/01/20210401_r_isobaricInhPa_500.grib2
2023-12-04 14:09     28262797  s3://meps-ai-data/meps/2021/04/01/20210401_r_isobaricInhPa_700.grib2
2023-12-04 14:09     31595239  s3://meps-ai-data/meps/2021/04/01/20210401_r_isobaricInhPa_850.grib2
2023-12-04 14:09     31899793  s3://meps-ai-data/meps/2021/04/01/20210401_r_isobaricInhPa_925.grib2
2023-12-04 14:09     28645578  s3://meps-ai-data/meps/2021/04/01/20210401_t_heightAboveGround_0.grib2
2023-12-04 14:09     26870453  s3://meps-ai-data/meps/2021/04/01/20210401_t_heightAboveGround_2.grib2
2023-12-04 14:09     25658079  s3://meps-ai-data/meps/2021/04/01/20210401_t_isobaricInhPa_1000.grib2
2023-12-04 14:09     27753625  s3://meps-ai-data/meps/2021/04/01/20210401_t_isobaricInhPa_300.grib2
2023-12-04 14:09     24468606  s3://meps-ai-data/meps/2021/04/01/20210401_t_isobaricInhPa_500.grib2
2023-12-04 14:09     25122048  s3://meps-ai-data/meps/2021/04/01/20210401_t_isobaricInhPa_700.grib2
2023-12-04 14:09     26344426  s3://meps-ai-data/meps/2021/04/01/20210401_t_isobaricInhPa_850.grib2
2023-12-04 14:09     26068047  s3://meps-ai-data/meps/2021/04/01/20210401_t_isobaricInhPa_925.grib2
2023-12-04 14:09     29232834  s3://meps-ai-data/meps/2021/04/01/20210401_u_heightAboveGround_10.grib2
2023-12-04 14:09     29302258  s3://meps-ai-data/meps/2021/04/01/20210401_u_isobaricInhPa_1000.grib2
2023-12-04 14:09     26144564  s3://meps-ai-data/meps/2021/04/01/20210401_u_isobaricInhPa_300.grib2
2023-12-04 14:09     26812729  s3://meps-ai-data/meps/2021/04/01/20210401_u_isobaricInhPa_500.grib2
2023-12-04 14:09     28568376  s3://meps-ai-data/meps/2021/04/01/20210401_u_isobaricInhPa_700.grib2
2023-12-04 14:09     28312641  s3://meps-ai-data/meps/2021/04/01/20210401_u_isobaricInhPa_850.grib2
2023-12-04 14:09     28337723  s3://meps-ai-data/meps/2021/04/01/20210401_u_isobaricInhPa_925.grib2
2023-12-04 14:09     31184313  s3://meps-ai-data/meps/2021/04/01/20210401_v_heightAboveGround_10.grib2
2023-12-04 14:09     30530973  s3://meps-ai-data/meps/2021/04/01/20210401_v_isobaricInhPa_1000.grib2
2023-12-04 14:09     25777399  s3://meps-ai-data/meps/2021/04/01/20210401_v_isobaricInhPa_300.grib2
2023-12-04 14:09     27320827  s3://meps-ai-data/meps/2021/04/01/20210401_v_isobaricInhPa_500.grib2
2023-12-04 14:09     28777624  s3://meps-ai-data/meps/2021/04/01/20210401_v_isobaricInhPa_700.grib2
2023-12-04 14:09     28526243  s3://meps-ai-data/meps/2021/04/01/20210401_v_isobaricInhPa_850.grib2
2023-12-04 14:09     28627889  s3://meps-ai-data/meps/2021/04/01/20210401_v_isobaricInhPa_925.grib2
2023-12-04 14:09     23774834  s3://meps-ai-data/meps/2021/04/01/20210401_z_isobaricInhPa_1000.grib2
2023-12-04 14:09     22727988  s3://meps-ai-data/meps/2021/04/01/20210401_z_isobaricInhPa_300.grib2
2023-12-04 14:09     21588744  s3://meps-ai-data/meps/2021/04/01/20210401_z_isobaricInhPa_500.grib2
2023-12-04 14:09     20765223  s3://meps-ai-data/meps/2021/04/01/20210401_z_isobaricInhPa_700.grib2
2023-12-04 14:09     23393451  s3://meps-ai-data/meps/2021/04/01/20210401_z_isobaricInhPa_850.grib2
2023-12-04 14:09     23373109  s3://meps-ai-data/meps/2021/04/01/20210401_z_isobaricInhPa_925.grib2
```

# Codes to download and preprocess MEPS data for AI training

* fetch-from-arcus.py
  * download grib files from MEtcoop arcus archive
  * modify metadata so that grib files look like 0-hour forecasts
* fetch-from-met-thredds.py
  * download netcdf from met.no thredds archive and convert to grib
  * modify metadata so that grib files look like 0-hour forecasts
* gridpp_analysis.py
  * correct meps forecasts with observations
  * uses gridpp from met.no (https://github.com/metno/gridpp)
* verify.sh
  * script to verify that the archive has all the required files for one day


## Usage examples

* For temperature
```
python gridpp_analysis.py --topography_data mnwc-Z-M2S2.grib2 --landseacover mnwc-LC-0TO1.grib2 --parameter_data mnwc-T-K.grib2 --output T-K.grib2 --parameter temperature
```  

## Authors
leila.hieta@fmi.fi mikko.partio@fmi.fi

## Known issues, features and development ideas
 
 
