import boto3
import time
import datetime
import os
import sys

if len(sys.argv) < 5:
    print(
        "Usage: python fetch-netatmo-from-glacier.py <year> <start-week-number> <stop-week-number> <destination-directory>"
    )
    print("stop-week-number is not included in the range")
    sys.exit(1)

year = int(sys.argv[1])
start_week_number = int(sys.argv[2])
stop_week_number = int(sys.argv[3])
destination_directory = sys.argv[4]

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    aws_session_token=os.environ["AWS_SESSION_TOKEN"],
)

bucket_name = "fmi-iot-obs-arch"
object_keys = []

for w in range(start_week_number, stop_week_number):
    for d in range(1, 8):
        object_keys.append(
            "{}/w{:02d}/{}_prod_id_3.csv".format(
                year, w, datetime.date.fromisocalendar(2021, w, d).isoformat()
            )
        )

print(bucket_name)
print(object_keys)
restore_request = {"Days": 2, "GlacierJobParameters": {"Tier": "Standard"}}

for object_key in object_keys:
    s3.restore_object(
        Bucket=bucket_name, Key=object_key, RestoreRequest=restore_request
    )

for object_key in object_keys:
    while True:
        print(
            "{} Checking if {} is restored".format(datetime.datetime.now(), object_key)
        )
        response = s3.head_object(Bucket=bucket_name, Key=object_key)
        if "Restore" in response and 'ongoing-request="false"' in response["Restore"]:
            break
        time.sleep(1800)  # Sleep 30mins

    local_file_path = destination_directory + "/" + os.path.basename(object_key)
    print("Downloading file {} to {}".format(object_key, local_file_path))
    s3.download_file(bucket_name, object_key, local_file_path)
