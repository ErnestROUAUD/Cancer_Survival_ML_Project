import os
import re
import boto3
import duckdb
from tqdm import tqdm
import requests
import time


# def RNA_seq_up_and_download(folder, file_id, s3_client):
#     s3_key = f"raw/RNA_seq/{folder}/{file_id}"

#     try: 
#         s3_client.head_object(Bucket = os.getenv("AWS_BUCKET_NAME"), Key = s3_key)
#         print(f"{file_id} already exist, skipped.")
#         return
#     except s3_client.exceptions.ClientError:
#         pass

#     file_API_endpt = f"https://api.gdc.cancer.gov/data/{file_id}"
#     response = requests.get(
#         file_API_endpt,
#         stream=True
#     )

#     if  response.status_code != 200: 
#         print(f"{file_id} cannot be downloaded: {response.status_code}")
#         return

#     #Stream into s3 directly 
#     s3_client.upload_fileobj(
#         response.raw, 
#         os.getenv("AWS_BUCKET_NAME"),
#         s3_key
#     )

#     print(f"done uploading: {s3_key}")


def RNA_seq_up_and_download(folder, file_id, s3_client, max_attempts=4):
    bucket = os.getenv("AWS_BUCKET_NAME")
    s3_key = f"raw/RNA_seq/{folder}/{file_id}"

    try:
        s3_client.head_object(Bucket=bucket, Key=s3_key)
        print(f"{file_id} already exist, skipped.")
        return "skipped"
    except s3_client.exceptions.ClientError:
        pass

    url = f"https://api.gdc.cancer.gov/data/{file_id}"

    for attempt in range(1, max_attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=(10, 60)) as response:
                if response.status_code != 200:
                    raise IOError(f"status {response.status_code}")
                expected = int(response.headers.get("Content-Length", 0))
                s3_client.upload_fileobj(response.raw, bucket, s3_key)

            uploaded = s3_client.head_object(Bucket=bucket, Key=s3_key)["ContentLength"]
            if expected and uploaded != expected:
                s3_client.delete_object(Bucket=bucket, Key=s3_key)
                raise IOError(f"size mismatch: {uploaded} vs {expected}")

            print(f"done uploading: {s3_key}")
            return "uploaded"

        except Exception as e:
            print(f"{file_id}: attempt {attempt}/{max_attempts} failed ({e})")
            if attempt < max_attempts:
                time.sleep(2 ** attempt)   # waits 2, 4, 8 seconds

    return "failed"


def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )


def setup_duckdb_s3():
    duckdb.sql("INSTALL httpfs; LOAD httpfs;")
    duckdb.sql(f"""
        SET s3_region='{os.getenv("AWS_REGION")}';
        SET s3_access_key_id='{os.getenv("AWS_ACCESS_KEY_ID")}';
        SET s3_secret_access_key='{os.getenv("AWS_SECRET_ACCESS_KEY")}';
    """)


def RNA_to_parquet(BUCKET, SUBFOLDER_PATH=None, selected_case_id=None, s3_client=None):
    if s3_client is None:
        s3_client = get_s3_client()
    setup_duckdb_s3()

    selected_case_id_set = set(selected_case_id) if selected_case_id else set()

    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    )

    Prefix = f"{SUBFOLDER_PATH}" if SUBFOLDER_PATH else ""
    FILE_NAME = SUBFOLDER_PATH.rstrip('/').split('/')[-1]

    paginator = s3_client.get_paginator('list_objects_v2')
    result = paginator.paginate(Bucket=BUCKET, Prefix=Prefix)
    all_keys = [key for page in result for key in page["Contents"]]

    print(f"{len(all_keys)} objects has been found under path: s3://{BUCKET}/{Prefix}")

    while True:
        user_input = input("Turn all objects into parquet file? y/n")

        if user_input == "y":
            table_created = False

            for key in tqdm(all_keys, desc="Loading RNA-seq files"):
                files_url = key["Key"]
                UUID = files_url.split('/')[-1]
                s3_path = f"s3://{BUCKET}/{Prefix}/{UUID}"

                if not uuid_pattern.match(UUID):
                    print(f"Skipping non-expression file: {files_url}")
                    continue
                if UUID not in selected_case_id_set:
                    continue

                query = f"""
                    SELECT gene_id, gene_name, unstranded, '{UUID}' AS file_id
                    FROM read_csv('{s3_path}', delim='\t', comment='#')
                    WHERE gene_id not in ('N_unmapped', 'N_multimapping', 'N_noFeature', 'N_ambiguous')
                    AND gene_type = 'protein_coding'
                """

                if not table_created:
                    duckdb.sql(f"CREATE OR REPLACE TABLE RNA_seq_expression_{FILE_NAME} AS {query}")
                    table_created = True
                else:
                    duckdb.sql(f"INSERT INTO RNA_seq_expression_{FILE_NAME} {query}")

            RNA_seq_expression = duckdb.sql(f"""
                SELECT * FROM RNA_seq_expression_{FILE_NAME}
            """).df()

            RNA_seq_expression.to_parquet(
                os.path.join(os.getenv("Output_path"), f'RNA_seq_expression_{FILE_NAME}.parquet')
            )
            break

        elif user_input == "n":
            print("aborted")
            break