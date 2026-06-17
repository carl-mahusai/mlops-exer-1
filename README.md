# mlops-exer-1

Development
1. Create your virtual environment. Note that this was made with Python 3.14.5

2. Open your CLI and run
```
pip install -r requirements/requirements-dev.txt
```

this will install the dev requirements

3. start mlflow at the root folder like so
```
mlflow server --port 5001 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./artifacts --serve-artifacts --allowed-hosts "host.docker.internal,host.docker.internal:*,http://host.docker.internal,http://host.docker.internal:*,localhost,localhost:*,http://localhost,http://localhost:*,http://127.0.0.1:*,http://127.0.0.1,127.0.0.1,127.0.0.1:*"
```

this will run mlflow server, create an sqlite file named mlflow.db for the backend store where runs metrics are stored and a folder named mlruns which will contain your experiment artifacts. ```--allowed-hosts``` will allow connections from the listed hosts. The listed hosts are for the black box docker container.

if you want to use postgresql for the backend-store-uri, you can do so like this
a. install ```psycopg2-binary```
```
pip install psycopg2-binary
```
change the owner of your mlflow db to your mlflow user
```
GRANT ALL PRIVILEGES ON DATABASE <mlflow_db> TO <mlflow_db_user>;
ALTER DATABASE <mlflow_db> OWNER TO <mlflow_db_user>;
```


change your mlflow server's backend-store-uri like this
```
mlflow server --port 5001 --backend-store-uri postgresql://<db_user>:<db_password>@<db_host>:<db_port>/<database_name> --artifacts-destination ./artifacts --serve-artifacts --allowed-hosts "host.docker.internal,host.docker.internal:*,http://host.docker.internal,http://host.docker.internal:*,localhost,localhost:*,http://localhost,http://localhost:*,http://127.0.0.1:*,http://127.0.0.1,127.0.0.1,127.0.0.1:*"
```

for my local setup, it would be like this
```
mlflow server --port 5001 --backend-store-uri postgresql://mlflow_metrics:WBev24h4@winhost:5432/mlflow_metrics --artifacts-destination ./artifacts --serve-artifacts --allowed-hosts "host.docker.internal,host.docker.internal:*,http://host.docker.internal,http://host.docker.internal:*,localhost,localhost:*,http://localhost,http://localhost:*,http://127.0.0.1:*,http://127.0.0.1,127.0.0.1,127.0.0.1:*"
```


you can view mlflow in
```
http://127.0.0.1:5001
```

4. Make sure you're at the root of the repo then run the following from your CLI to initiate training
```
python -m training.run_training --data=<path to input data> --name_of_label_column=<column where the labels are> --name_of_message_column=<name of column where the messages are> --mlflow_tracking_uri='http://127.0.0.1:5001'
```

for example
```
python -m training.run_training --data='test_dataset/spam.csv' --name_of_label_column='v1' --name_of_message_column='v2' --mlflow_tracking_uri='http://127.0.0.1:5001'
```

you can run with distributed processing by adding the ```--distributed_processing``` argument
```
python -m training.run_training --data='test_dataset/spam.csv' --name_of_label_column='v1' --name_of_message_column='v2' --mlflow_tracking_uri='http://127.0.0.1:5001' --max_epoch=20 --distributed_processing
```

but it may crash when using mllflow with an sqlite db.

Notes regarding the training script
1. The label column should contain "ham" for non-spam messages and "spam" for spam messages. you may add other columns aside from the label and messages column but those would be ignored
2. The optional commands are
   - --batch_size. default is 64
   - --max_epochs. default is 2
3. Training has early stopping in place where it will stop after 3 validation checks with no decrease in training loss
3. --mlflow_tracking_uri is optional. if it's ommited, the vocab file and checkpoint file are saved at the root. with mlflow server running and --mlflow_tracking_uri included in the arguments, this will upload the vocab file and checkpoint file to your local mlflow
4. During testing, when i was using distributed processing with mlflow tracking, the part which saves artifacts to mlflow kept crashing even if i had everything setup so that only rank 0 will handle saving. it does sometimes work but just in case you're having issues with training using mlflow, don't include this argument. You can run distributed training consistently if you remove ```--mlflow_tracking_uri``` while ```--distributed_processing``` is on although this will save the final checkpoint and vocab files at the root and metrics won't be recorded. you can try to keep running the training until it works. Note that when turned on, it will use the gpu for training. This error still happens even when using postgresql which can handle distributed processing. Probably due to how artifacts are saved in the artifacts folder but i couldn't confirm with a local setup since i'm using a local store. this will probably work better with an s3 setup


5. Build the docker file for the predictor
stay on the parent directory and run the following docker build command. 
```
docker build -f deployment/Dockerfile -t spam-prediction-service-mlflow:v1 .
```

to run the container locally
```
docker run -it --rm -p 9696:9696 -e RUN_ID=<run id in mlflow> -e TRACKING_URI=<tracking uri used by mlflow> spam-prediction-service-mlflow:v1
```

for a local run, add ```--add-host=host.docker.internal:host-gateway``` and use "http://host.docker.internal:5001" for the tracking uri. this will connect to the mlflow setup running 
```
docker run -it --rm -p 9696:9696 -e RUN_ID=3096bcb13e084fa2a52dbf1a55620e6f -e TRACKING_URI="http://host.docker.internal:5001" --add-host=host.docker.internal:host-gateway spam-prediction-service-mlflow:v1
```

to test that the blackbox container running locally can connect to your mlflow setup that's also running locally, run the following in the container
```
curl http://host.docker.internal:5001
```

check your local mlflow server. it should return a 200 response

to test the docker container, from the root folder, run this in your CLI
```
python deployment/test.py
```