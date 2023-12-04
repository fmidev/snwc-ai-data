#!/bin/bash
# This is a simple script to store ML realtime forecasts for testing

# Script to run obs analysis for grib2 data (S3://meps-ai-data)

python3 --version

#WEEKDAY=`date +"%a"`
#HOD=`date +"%H"`
#AIKA1=`date "+%Y%m%d%H"  -u`
#AIKA=$(echo "$( date "+%Y%m%d%H"  -u)")

# Check if the number of hours is provided as a command-line argument
if [ $# -eq 0 ]; then
  echo "Usage: $0 <hours>"
  exit 1
fi

# Input hours
hours=$1

# Calculate the new time
NN=$(date -u -d "$hours hours ago" "+%Y%m%d%H")

echo "New time: $NN"

# laske piste-ennusteet kaikille parametreille
# T2m
#python3 gridpp_analysis.py --topography_data s3://routines-data/mnwc-biascorrection/preop/"$NN"00/Z-M2S2.grib2 --landseacover s3://routines-data/mnwc-biascorrection/preop/"$NN"00/LC-0TO1.grib2 --parameter_data s3://routines-data/mnwc-biascorrection/preop/"$NN"00/T-K.grib2 --output poista_T2.grib2 --parameter temperature
# RH
python3 gridpp_analysis.py --topography_data s3://routines-data/mnwc-biascorrection/preop/"$NN"00/Z-M2S2.grib2 --landseacover s3://routines-data/mnwc-biascorrection/preop/"$NN"00/LC-0TO1.grib2 --parameter_data s3://routines-data/mnwc-biascorrection/preop/"$NN"00/RH-0TO1.grib2 --output poista_RH.grib2 --parameter humidity
# WS
#python3 gridpp_analysis.py --topography_data s3://routines-data/mnwc-biascorrection/preop/"$NN"00/Z-M2S2.grib2 --landseacover s3://routines-data/mnwc-biascorrection/preop/"$NN"00/LC-0TO1.grib2 --parameter_data s3://routines-data/mnwc-biascorrection/preop/"$NN"00/FF-MS.grib2 --output poista_WS.grib2 --parameter windspeed
# WG
#python3 gridpp_analysis.py --topography_data s3://routines-data/mnwc-biascorrection/preop/"$NN"00/Z-M2S2.grib2 --landseacover s3://routines-data/mnwc-biascorrection/preop/"$NN"00/LC-0TO1.grib2 --parameter_data s3://routines-data/mnwc-biascorrection/preop/"$NN"00/FFG-MS.grib2 --output poista_WG.grib2 --parameter gust

