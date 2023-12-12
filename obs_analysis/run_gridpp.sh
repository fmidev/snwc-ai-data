#!/bin/bash
# Script to run obs analysis for grib2 data (S3://meps-ai-data)

python3 --version

timedate=$1
parameter=$2
echo $timedate # 20220101
echo $parameter # t=temperature, r=humidity, uv=windspeed, fg=gust
year=${timedate:0:4}
month=${timedate:4:2}
day=${timedate:6:2}
#wrkdir=/home/hietal/Desktop/Python_projects/dev-snwc-ai
wrkdir=/home/users/hietal/statcal/python_projects/snwc-ai-data/obs_analysis
cd $wrkdir
pwd
start_time=$(date +%s)

if [ "$parameter" == "t" ]; then
  s3cmd get s3://meps-ai-data/meps/"$year"/"$month"/"$day"/"$timedate"_t_heightAboveGround_2.grib2
  python3.9 "$wrkdir"/gridpp_analysis.py --topography_data Z-M2S2.grib2 --landseacover LC-0TO1.grib2 --parameter_data "$timedate"_t_heightAboveGround_2.grib2 --output poista_T2.grib2 --parameter t
elif [ "$parameter" == "uv" ]; then
  s3cmd get s3://meps-ai-data/meps/"$year"/"$month"/"$day"/"$timedate"_u_heightAboveGround_10.grib2
  s3cmd get s3://meps-ai-data/meps/"$year"/"$month"/"$day"/"$timedate"_v_heightAboveGround_10.grib2
  python3.9 "$wrkdir"/gridpp_analysis.py --topography_data Z-M2S2.grib2 --landseacover LC-0TO1.grib2 --parameter_data "$timedate"_u_heightAboveGround_10.grib2 --v_component "$timedate"_v_heightAboveGround_10.grib2 --output poista_U.grib2 --output_v poista_V.grib2 --parameter uv
elif [ "$parameter" == "fg" ]; then
  s3cmd get s3://meps-ai-data/meps/"$year"/"$month"/"$day"/"$timedate"_fg_heightAboveGround_10.grib2
  python3.9 "$wrkdir"/gridpp_analysis.py --topography_data Z-M2S2.grib2 --landseacover LC-0TO1.grib2 --parameter_data "$timedate"_fg_heightAboveGround_10.grib2 --output poista_WG.grib2 --parameter fg
elif [ "$parameter" == "r" ]; then
  s3cmd get s3://meps-ai-data/meps/"$year"/"$month"/"$day"/"$timedate"_r_heightAboveGround_2.grib2
  python3.9 "$wrkdir"/gridpp_analysis.py --topography_data Z-M2S2.grib2 --landseacover LC-0TO1.grib2 --parameter_data "$timedate"_r_heightAboveGround_2.grib2 --output poista_RH.grib2 --parameter r
else
  echo "parameter not found"
  exit 1
fi

#rm "$timedate"_*
# Measure end time
end_time=$(date +%s)
# Calculate execution time
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"


