#!/usr/bin/env python3

import os
import boto3
import datetime
import json
import eccodes as ecc
from botocore.config import Config
import sys
import argparse
from io import BytesIO


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
    # parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    args.leadtimes = list(map(lambda x: int(x), args.leadtimes.split(",")))
    args.level_values = list(map(lambda x: int(x), args.level_values.split(",")))

    if type(args.level_values) != list:
        args.level_values = [args.level_values]
    return args


args = parse_args()


def param_to_id(param):
    # (discipline, parameterCategory, parameterNumber, typeOfStatisticalProcessing)
    params = {
        "t": (0, 0, 0, "NULL"),
        "fg": (0, 2, 22, 2),
        "mld": (0, 19, 3, "NULL"),
        "pres": (0, 3, 0, "NULL"),
        "r": (0, 1, 192, "NULL"),
        "u": (0, 2, 2, "NULL"),
        "v": (0, 2, 3, "NULL"),
        "z": (0, 3, 4, "NULL"),
        "tcc": (0, 6, 192, "NULL"),
    }

    try:
        return params[param]
    except KeyError as e:
        raise Exception(f"Param {param} not recognized")


def level_to_id(level):
    levels = {
        "heightAboveGround": 103,
        "heightAboveSea": 102,
        "hybrid": 105,
        "isobaricInhPa": 100,
    }

    try:
        return levels[level]
    except KeyError as e:
        raise Exception(f"Level {level} not recognized")


def init():
    global bucket, s3

    bucket = "calibration"

    config = Config(
        retries={"max_attempts": 3, "mode": "standard"},
        region_name="us-east-1",
    )
    s3 = boto3.resource(
        "s3",
        config=config,
        endpoint_url="https://arcus-s3.nsc.liu.se",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def get_index(date):
    key = "MEPS_prod/{}/index.json".format(date.strftime("%Y/%m/%d/%H"))
    f = s3.Object(bucket, key).get()["Body"]

    return json.loads(f.read())


def get_grib(key, offset, length):
    f = s3.Object(bucket, key).get(
        Range="bytes={}-{}".format(offset, offset + length - 1)
    )["Body"]
    gid = ecc.codes_new_from_message(f.read())

    return gid


def modify_grib(grib):
    """
    Set step to zero and dataTime to step
    """

    hour = ecc.codes_get(grib, "hour")
    ft = ecc.codes_get(grib, "forecastTime")

    tosp = None

    try:
        tosp = ecc.codes_get(grib, "typeOfStatisticalProcessing")
    except ecc.KeyValueNotFoundError as e:
        pass

    if tosp == 2:
        # gust

        # gust at step=0 has invalid time period in arcus
        # (0 hours), fix that

        lotr = ecc.codes_get(grib, "lengthOfTimeRange")

        ecc.codes_set(grib, "forecastTime", -1)
        ecc.codes_set(grib, "lengthOfTimeRange", 1)
        ecc.codes_set(grib, "hour", hour + ft + lotr)

        return grib

    ecc.codes_set(grib, "hour", ft + hour)
    ecc.codes_set(grib, "forecastTime", 0)

    return grib


def fetch():
    init()

    date = datetime.datetime(args.year, args.month, args.day, args.cycle)
    idx = get_index(date)

    edition = 2
    typeOfFirstFixedSurface = level_to_id(args.level)
    perturbationNumber = args.perturbation_number
    (
        discipline,
        parameterCategory,
        parameterNumber,
        typeOfStatisticalProcessing,
    ) = param_to_id(args.param)

    myio = []
    for i in range(len(args.leadtimes)):
        myio.append([])
        for _ in args.level_values:
            myio[i].append(BytesIO())

    for i, lt in enumerate(args.leadtimes):
        for j, level in enumerate(args.level_values):
            key, offset, length = tuple(
                idx[
                    "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
                        edition,
                        lt,
                        typeOfFirstFixedSurface,
                        level,
                        perturbationNumber,
                        discipline,
                        parameterCategory,
                        parameterNumber,
                        typeOfStatisticalProcessing,
                    )
                ]
            )

            grib = get_grib(key, offset, length)
            grib = modify_grib(grib)

            ecc.codes_write(grib, myio[i][j])
            ecc.codes_release(grib)

    return myio


def create_filename():

    if args.output_filename is not None:
        return "{}/{}".format(args.output_dir, args.output_filename)

    if args.merge:
        filename = (
            "{}/{}/{:02d}/{:02d}/raw/{}{:02d}{:02d}{:02d}_{}_{}_mbr{}.grib2".format(
                args.output_dir,
                args.year,
                args.month,
                args.day,
                args.year,
                args.month,
                args.day,
                args.cycle,
                args.param,
                args.level,
                args.perturbation_number,
            )
        )

    else:
        filename = (
            "{}/{}/{:02d}/{:02d}/raw/{}{:02d}{:02d}{:02d}_{}_{}_{}_mbr{}.grib2".format(
                args.output_dir,
                args.year,
                args.month,
                args.day,
                args.year,
                args.month,
                args.day,
                args.cycle + args.leadtimes[i],
                args.param,
                args.level,
                args.level_values[j],
                args.perturbation_number,
            )
        )

    return filename


def write(data):

    for i in range(len(data)):
        for j in range(len(data[i])):
            filename = create_filename()

            dirname = os.path.dirname(filename)
            os.makedirs(dirname, exist_ok=True)

            open_mode = "ab" if args.merge else "wb"
            with open(filename, open_mode) as f:
                f.write(data[i][j].getbuffer())

            print(f"Wrote to file {filename}")


data = fetch()
write(data)
