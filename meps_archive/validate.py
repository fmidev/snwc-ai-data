#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np
from eccodes import *
import argparse
from pathlib import Path

# Parameter validation ranges and settings
# Format: parameter_name: {
#   'min_valid': minimum valid value,
#   'max_valid': maximum valid value,
#   'allow_missing': whether missing values are allowed,
#   'allow_zero_std': whether zero standard deviation is allowed
# }
PARAMETER_LIMITS = {
    # Cloud parameters
    "cape": {
        "min_valid": 0,
        "max_valid": 10000,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # J/kg
    "cdcb": {
        "min_valid": 0,
        "max_valid": 29000,
        "allow_missing": True,
        "allow_zero_std": False,
    },  # m (cloud base)
    "cdct": {
        "min_valid": 0,
        "max_valid": 29000,
        "allow_missing": True,
        "allow_zero_std": False,
    },  # m (cloud top)
    "cldice": {
        "min_valid": -1e-3,
        "max_valid": 1,
        "allow_missing": False,
        "allow_zero_std": True,
    },  # kg/kg
    "cldwat": {
        "min_valid": -1e-3,
        "max_valid": 0.01,
        "allow_missing": False,
        "allow_zero_std": True,
    },  # kg/kg
    "fg": {
        "min_valid": 0,
        "max_valid": 50,
        "allow_missing": False,
        "allow_zero_std": False,
    },
    "hcc": {
        "min_valid": 0,
        "max_valid": 1.0001,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # fraction
    "lcc": {
        "min_valid": 0,
        "max_valid": 1.0001,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # fraction
    "mcc": {
        "min_valid": 0,
        "max_valid": 1.0001,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # fraction
    "tcc": {
        "min_valid": 0,
        "max_valid": 1.0001,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # fraction
    "cc": {
        "min_valid": 0,
        "max_valid": 1.0001,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # fraction
    # Precipitation parameters
    "graupelmr": {
        "min_valid": -1e-3,
        "max_valid": 0.01,
        "allow_missing": False,
        "allow_zero_std": True,
    },  # kg/kg
    "mld": {
        "min_valid": 0,
        "max_valid": 29000,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # m (mixed layer depth)
    "pres": {
        "min_valid": 18000,
        "max_valid": 110000,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # Pa
    "rainmr": {
        "min_valid": -1e-6,
        "max_valid": 0.01,
        "allow_missing": False,
        "allow_zero_std": True,
    },  # kg/kg
    "snowmr": {
        "min_valid": -1e-6,
        "max_valid": 0.01,
        "allow_missing": False,
        "allow_zero_std": True,
    },  # kg/kg
    "tp": {
        "min_valid": -1e-3,
        "max_valid": 66,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # mm/h
    "ptype": {
        "min_valid": 0,
        "max_valid": 8,
        "allow_missing": True,
        "allow_zero_std": False,
    },  # categorical
    # Atmospheric state parameters
    "q": {
        "min_valid": -1e-3,
        "max_valid": 0.1,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # kg/kg (specific humidity)
    "r": {
        "min_valid": -1e-3,
        "max_valid": 1.0035,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # % (relative humidity)
    "t": {
        "min_valid": 150,
        "max_valid": 350,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # K (temperature)
    "z": {
        "min_valid": -5000,
        "max_valid": 350000,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # m (geopotential height)
    # Wind parameters
    "u": {
        "min_valid": -200,
        "max_valid": 200,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # m/s
    "v": {
        "min_valid": -200,
        "max_valid": 200,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # m/s
    "w": {
        "min_valid": -50,
        "max_valid": 50,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # m/s
    # Energy parameters
    "tke": {
        "min_valid": 0,
        "max_valid": 1000,
        "allow_missing": False,
        "allow_zero_std": True,
    },  # J/kg
    "ssrd": {
        "min_valid": 0,
        "max_valid": 50000000,
        "allow_missing": False,
        "allow_zero_std": True,
    },  # J/m^2
    "strd": {
        "min_valid": 0,
        "max_valid": 50000000,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # J/m^2
    # Other parameters
    "li": {
        "min_valid": 0,
        "max_valid": 3500,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # K (lifted index)
    "sf": {
        "min_valid": -1e-3,
        "max_valid": 23,
        "allow_missing": False,
        "allow_zero_std": True,
    },  # m (snowfall)
    "vis": {
        "min_valid": 0,
        "max_valid": 100000,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # m (visibility)
    "tcwv": {
        "min_valid": 0,
        "max_valid": 100,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # kg/m^2 (total column water vapor)
    "tcw": {
        "min_valid": 0,
        "max_valid": 100,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # kg/m^2 (total column water vapor)
    "h": {
        "min_valid": 8,
        "max_valid": 31000,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # m (height above ground)
    "preform": {
        "min_valid": 0,
        "max_valid": 5,
        "allow_missing": False,
        "allow_zero_std": False,
    },  # codetable (preform)
}


def extract_parameter_from_filename(filename):
    """Extract parameter name from GRIB2 filename."""
    parts = filename.split("_")
    if len(parts) >= 2:
        return parts[1]  # Parameter is the second part after date
    return None


def validate_grib_file(filepath, datadate):
    filename = os.path.basename(filepath)
    parameter = extract_parameter_from_filename(filename)

    if parameter not in PARAMETER_LIMITS:
        return {
            "filename": filename,
            "parameter": parameter,
            "status": "ERROR",
            "message": f"Unknown parameter: {parameter}",
            "stats": {},
        }

    limits = PARAMETER_LIMITS[parameter]
    required_message_count = 24  # Expecting 24 messages for each day (hourly data)
    required_datetime_sum = (
        27600  # Sum of dataTime values for 24 hours (0+100+200+...+2300)
    )

    if "_hybrid_" in filename:
        required_message_count = 24 * 65  # 65 vertical levels for hybrid files
        required_datetime_sum = 27600 * 65

    try:
        count = 0
        datatimesum = 0
        levelvaluesum = 0

        with open(filepath, "rb") as f:
            errors = []

            while True:
                gid = codes_grib_new_from_file(f)

                if gid is None:
                    break

                count += 1
                datatimesum += int(codes_get(gid, "dataTime"))
                levelvaluesum += int(codes_get(gid, "level"))

                assert int(codes_get(gid, "dataDate")) == int(datadate), (
                    f"Data date mismatch in {filename}: expected {args.datadate}, "
                    f"found {codes_get(gid, 'dataDate')}"
                )

                # After
                values = codes_get_values(gid)
                missing_value = codes_get(gid, "missingValue")

                # Create mask for missing values (both NaN and the specific missing value)
                missing_mask = np.isnan(values) | (values == missing_value)
                missing_count = np.sum(missing_mask)
                valid_values = values[~missing_mask]

                if len(valid_values) == 0:
                    codes_release(gid)
                    return {
                        "filename": filename,
                        "parameter": parameter,
                        "status": "ERROR",
                        "message": "All values are missing",
                        "stats": {},
                    }

                min_val = np.min(valid_values)
                max_val = np.max(valid_values)
                std_val = np.std(valid_values)
                mean_val = np.mean(valid_values)

                stats = {
                    "min": min_val,
                    "max": max_val,
                    "mean": mean_val,
                    "std": std_val,
                    "missing_count": missing_count,
                    "total_count": len(values),
                }

                # Check minimum value
                if min_val < limits["min_valid"]:
                    errors.append(
                        f'Minimum value {min_val:.7f} below valid range ({limits["min_valid"]})'
                    )

                # Check maximum value
                if max_val > limits["max_valid"]:
                    errors.append(
                        f'Maximum value {max_val:.7f} above valid range ({limits["max_valid"]})'
                    )

                # Check missing values
                if missing_count > 0 and not limits["allow_missing"]:
                    errors.append(
                        f"Found {missing_count} missing values (not allowed for this parameter)"
                    )

                # Check standard deviation
                if std_val == 0 and not limits["allow_zero_std"]:
                    errors.append(
                        "Standard deviation is zero (not allowed for this parameter)"
                    )

                codes_release(gid)

            if count != required_message_count:
                return {
                    "filename": filename,
                    "parameter": parameter,
                    "status": "ERROR",
                    "message": f"Expected {required_message_count} messages, found {count}",
                    "stats": {},
                }

            if datatimesum != required_datetime_sum:
                return {
                    "filename": filename,
                    "parameter": parameter,
                    "status": "ERROR",
                    "message": f"Data time sum mismatch: expected {required_datetime_sum}, found {datatimesum}",
                    "stats": {},
                }

            if "_hybrid_" in filename and levelvaluesum != (
                24 * 2145
            ):  # sum(range(1, 66)) = 2145
                return {
                    "filename": filename,
                    "parameter": parameter,
                    "status": "ERROR",
                    "message": f"Level value sum mismatch for hybrid file: expected {24 * 2145}, found {levelvaluesum}",
                    "stats": {},
                }

            if errors:
                return {
                    "filename": filename,
                    "parameter": parameter,
                    "status": "ERROR",
                    "message": "; ".join(errors),
                    "stats": stats,
                }
            else:
                return {
                    "filename": filename,
                    "parameter": parameter,
                    "status": "OK",
                    "message": "All validations passed",
                    "stats": stats,
                }

    except Exception as e:
        return {
            "filename": filename,
            "parameter": parameter,
            "status": "ERROR",
            "message": f"Exception during validation: {str(e)}",
            "stats": {},
        }


def main():
    parser = argparse.ArgumentParser(description="Validate GRIB2 files in a directory")
    parser.add_argument("directory", help="Directory containing GRIB2 files")
    parser.add_argument(
        "--datadate",
        "-d",
        type=str,
        required=True,
        help="Data date in YYYYMMDD format (e.g., 20231001)",
    )
    parser.add_argument(
        "--num_gribs",
        "-n",
        type=int,
        default=179,
        help="Expected number of GRIB2 files",
    )  # 145 for refill
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        sys.exit(1)

    # Find all GRIB2 files
    grib_pattern = os.path.join(args.directory, "*.grib2")
    grib_files = glob.glob(grib_pattern)

    if not grib_files:
        print(f"No GRIB2 files found in directory '{args.directory}'")
        sys.exit(1)

    if len(grib_files) != args.num_gribs:
        print(
            f"Warning: Expected {args.num_gribs} GRIB2 files, found {len(grib_files)}. "
            "Please ensure all files are present."
        )
        sys.exit(1)

    print(f"Found {len(grib_files)} GRIB2 files to validate")
    print("=" * 60)

    results = []
    error_count = 0
    warning_count = 0

    for grib_file in sorted(grib_files):
        result = validate_grib_file(grib_file, args.datadate)
        results.append(result)

        if result["status"] == "ERROR":
            error_count += 1
            print(f"❌ {result['filename']}")
            print(f"  Parameter: {result['parameter']}")
            print(f"  Message: {result['message']}")
            if result["stats"] and args.verbose:
                stats = result["stats"]
                print(
                    f"  Stats: min={stats.get('min', 'N/A'):.6f}, max={stats.get('max', 'N/A'):.6f}, "
                    f"mean={stats.get('mean', 'N/A'):.6f}, std={stats.get('std', 'N/A'):.6f}, "
                    f"missing={stats.get('missing_count', 'N/A')}"
                )
            print()

        else:
            print(f"✅ {result['filename']} ({result['parameter']})")

            if args.verbose:
                if result["stats"]:
                    stats = result["stats"]
                    print(
                        f"  Stats: min={stats['min']:.6f}, max={stats['max']:.6f}, "
                        f"mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                        f"missing={stats['missing_count']}"
                    )

    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {len(results)}")
    print(f"Successful validations: {len(results) - error_count - warning_count}")
    print(f"Errors: {error_count}")

    if error_count == 0:
        print("\n✅ All validations passed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ {error_count} files failed validation")
        sys.exit(1)


if __name__ == "__main__":
    main()
