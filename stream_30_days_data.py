import requests
import json
import pandas as pd
import google.auth
import google.auth.transport.requests


REGION = "GET YOU VERTEX REGION"
ENDPOINT_ID = "ENDPOINT ID PROVIDED BY THE OUTPUT OF THE LAST STAGE OF THE PIPELINE"
PROJECT_ID = "YOU GCP PROJECT ID"

url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"
credentials, project_id = google.auth.default(
    scopes=[
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/cloud-platform.read-only",
    ]
)

instances = {"instances": []}
df = pd.read_csv("https://www.openml.org/data/get_csv/21792853/dataset")
expensive_df = df[df["price"] > 10000].sort_values("price", ascending=False)
df = df[df["price"] < 10000]

count = 5
chunk_size = 500
reset_index = True
min_chunk, max_chunk = 0, chunk_size
while count:
    print(count)
    print(f"Uploading data from: {str(pd.Timestamp.now() - pd.Timedelta(count, 'd'))}")
    if count < 10:
        print(expensive_df.iloc[min_chunk:max_chunk]["price"].mean())
        if reset_index:
            min_chunk, max_chunk = 0, 500
            reset_index = False
        for row_tuple in expensive_df.iloc[min_chunk:max_chunk].iterrows():
            row_dict = row_tuple[1].drop("price").to_dict()
            row_dict["record_id"] = row_tuple[1].name
            row_dict["ts"] = str(pd.Timestamp.now() - pd.Timedelta(count, "d"))
            instances["instances"].append(row_dict)
    else:
        print(expensive_df.iloc[min_chunk:max_chunk]["price"].mean())
        for row_tuple in df.iloc[min_chunk:max_chunk].iterrows():
            row_dict = row_tuple[1].drop("price").to_dict()
            row_dict["record_id"] = row_tuple[1].name
            row_dict["ts"] = str(pd.Timestamp.now() - pd.Timedelta(count, "d"))
            instances["instances"].append(row_dict)

    request = google.auth.transport.requests.Request()
    credentials.refresh(request)
    token = credentials.token
    headers = {"Authorization": "Bearer " + token}
    response = requests.post(url, json=instances, headers=headers)
    # print(response.text)
    print("---" * 15)
    instances["instances"] = []
    count -= 1
    min_chunk += chunk_size
    max_chunk += chunk_size
