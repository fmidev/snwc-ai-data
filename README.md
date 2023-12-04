# Codes to download and preprocess MEPS data for AI training
Code to add observation information to MEPS surface parameters: temperature, wind speed, wind gust and relative humidity (2m) using https://github.com/metno/gridpp   

## Usage
* For temperature
```
python gridpp_analysis.py --topography_data mnwc-Z-M2S2.grib2 --landseacover mnwc-LC-0TO1.grib2 --parameter_data mnwc-T-K.grib2 --output T-K.grib2 --parameter temperature
```  

## Authors
leila.hieta@fmi.fi mikko.partio@fmi.fi

## Known issues, features and development ideas
 
 
