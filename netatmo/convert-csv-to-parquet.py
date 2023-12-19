import json
import sys
import csv
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm

def read_json_file(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file {file_path}: {e}")


def read_csv_file(file_path, mid):
    try:
        with open(file_path, "r", newline="") as file:
            reader = csv.DictReader(file)
            data_list = []
            for row in tqdm(reader):
                if int(row["mid"]) == mid:
                    data_list.append(row)

            return data_list
    except FileNotFoundError:
        print(f"File not found: {file_path}")


def rearrange(data):
    stations = {}
    for station in data:
        stations[station["station_id"]] = station
    return stations


def merge(stations, observations):
    for o in tqdm(observations):
        station = stations.get(int(o["station_id"]))
        if station:
            o["latitude"] = station["geom"]["coordinates"][1]
            o["longitude"] = station["geom"]["coordinates"][0]
            o["altitude"] = station["altitude"]
    return observations


if len(sys.argv) != 4:
    print(
        "Usage: extract-temperature.py <observations.csv> <stations.json> <output.parquet>"
    )
    sys.exit(1)

print("Reading stations...")
stations = read_json_file(sys.argv[2])
stations = rearrange(stations)

print("Reading observations...")
observations = read_csv_file(sys.argv[1], 37)

print("Merging stations and observations...")
observations = merge(stations, observations)

df = pd.DataFrame.from_dict(observations)
df["data_time"] = pd.to_datetime(df["data_time"])
df["data_value"] = df["data_value"].astype("float32")
df["station_id"] = df["station_id"].astype("uint32")
df["data_quality"] = df["data_quality"].replace("", -1).astype("int8")
df = df.rename(
    columns={"data_time": "utctime", "data_value": "obs_value", "altitude": "elevation"}
)
df = df.drop(columns=["mid"])
df = df.sort_values(by=["utctime", "station_id"])
df = df.reset_index(drop=True)
table = pa.Table.from_pandas(df)

pq.write_table(table, sys.argv[3])
print("Wrote parquet file to " + sys.argv[3])
