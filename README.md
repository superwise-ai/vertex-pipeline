# ml-pipeline
Simple pipeline flow, using [Diamonds Dataset](https://www.kaggle.com/datasets/shivam2503/diamonds) to demonstrate Continuous training paradigm.


Prerequisites:
- gcloud installed
- GCP project & Storage bucket
- relevant gcloud permission  
	```sh 
	gcloud services enable compute.googleapis.com \ 
		containerregistry googleapis.com  \
		aiplatform.googleapis.com  \
		cloudbuild.googleapis.com \
		cloudfunctions.googleapis.com
	```
- Superwise Account
- following environment variables:
	- REGION - GCP region
	- PROJECT_ID - GCP project id ( can be retrieved using `gcloud config list --format 'value(core.project)'` )
	- BUCKET_NAME - name of GCS bucket
	- SUPERWISE_CLIENT_ID - retrieved from Superwise's platform
	- SUPERWISE_SECRET - retrieved from Superwise's platform

Usage:

`python3 -m venv venv && . ./venv/bin/activate && pip install requirements.txt`

`python pipeline.py`

Login into Google Vertex Console -> pipelines
