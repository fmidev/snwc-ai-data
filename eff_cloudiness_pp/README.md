## NWCSAF Effective cloudiness based total cloud cover analysis for MEPS domain
[NWCSAF Effective cloudiness](https://www.nwcsaf.org/ctth_description#2.-%20CTTH%20algorithm%20summary%20description) based total cloud cover (tcc) data is used to produce the operational tcc nowcast [Cloudcast](https://github.com/fmidev/cloudcast/tree/main) at FMI. 
Since the geostationary satellite-based product has quality issues at northern latitudes, additional processing has been applied to improve data quality. Similar processing, as used in Cloudcast production, is now applied to archive data to create a TCC analysis that can be used as training data.

#### Output data specifications
- Domain & resolution: MEPS (MEPS25D) northern europe in lambert conformal conic projection, 2.5 km grid
- Data coverage: 2021-09-15 - 2024-09-30 (contains missing timesteps, 685 timesteps/hours in total!)
- Temporal resolution: 1h
- tcc data range: 1...100

#### Data quality issues & processing
Data sources used to correct NWCSAF effective cloudiness data

| Data | Start date | End Date | Temporal resolution | Extra info |
| --- | --- | --- | --- | --- |
| NWCSAF effective cloudiness | 2018-11-01T00:00  | 2024-09-30T23:45 | 15min | FMI archive |
| NWCSAF cloud top temperature | 2018-11-01T00:00 | 2024-09-30T23:45 | 15min | FMI archive |
| MEPS low, medium, high cloud cover | 2021-09-14T22:00 | 2024-12... | 1h | Arcus archive |
| MEPS 2m relative humidity | 2021-04-01T00:00  | 2024-12... | FMI archive, data corrected by synop observations |
| ECMWF short wave radiation | 2020-02-05T00:00  | 2024-10-01T00:00 | 1h | FMI archive |
| Clear sky value of short wave radiation | | | 1h | FMI archive, Data calculated using pvlib Python package |

- There are gaps/missing values in the effective cloudiness data. These gaps are filled by examining the values of nearby grid points and adjusting the cloudiness accordingly.
- High level clouds have 100% value, but end users interpret the weather often as sunny/partly sunny. MEPS cloud layer information is used to reduce tcc values, if only high level clouds are present. 
- To improve cloud mask information, cloud cover is added if NWCSAF cloud top temperature has values but effective cloudiness is zero.
- Stratus clouds are sometimes falsely interpreted as clear sky in effective cloudiness. This is corrected by increasing tcc value if 2m relative humidity has high values.
- Between different cloud layers, there may be a narrow band of clear sky areas in effective cloudiness, possibly caused by shadow. We try to correct this by the methods mentioned above.
- Cumulus clouds are often interpreted as 100% tcc in effective cloudiness, possibly due to coarse resolution. The ratio of short wave radiation to clear sky value of short wave radiation is used to find the areas where there are cumulus clouds and from which the tcc values are reduced. This correction is effective during months April to September and during 7-17utc.  

#### Known issues in the data
- Since MEPS domain is on the very edge of the geostationary satellite's measurement area, the quality/resolution of the effective cloudiness data deteriorates further north, especially in the northeast corner where the quality is very poor.
- There are artifacts, such as single timesteps when most of the effective cloudiness field gets value of 100% often related to sunrise/sunset.
- The corrections are not perfect. While they reduce the errot from 100% to 50% in most cases, they may occasionally degrade data quality.
- Since some corrections are only applied during specific months or times of day, there is jumpiness in the data. 
     
#### Example



&nbsp;

&nbsp;
