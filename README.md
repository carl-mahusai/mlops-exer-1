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

you can view mlflow in
```
http://127.0.0.1:5001
```

4. Make sure you're at the root of the repo then run the following from your CLI to initiate training
```
python -m training.run_training --data=<path to input data> --name_of_label_column=<column where the labels are> --name_of_message_column=<name of column where the messages are> --mlflow_tracking_uri=''http://127.0.0.1:5001'
```

for example
```
python -m training.run_training --data='test_dataset/spam.csv' --name_of_label_column='v1' --name_of_message_column='v2' --mlflow_tracking_uri=''http://127.0.0.1:5001'
```

you can run with distributed processing by adding the ```--distributed_processing``` argument
```
python -m training.run_training --data='test_dataset/spam.csv' --name_of_label_column='v1' --name_of_message_column='v2' --mlflow_tracking_uri=''http://127.0.0.1:5001' --max_epoch=20 --distributed_processing
```

```
python -m training.run_training --data='test_dataset/spam.csv' --name_of_label_column='v1' --name_of_message_column='v2' --mlflow_tracking_uri='http://127.0.0.1:5001' --max_epoch=20 --distributed_processing
```

but it may crash when using mllflow with an sqlite db. Notes below

Notes regarding the training script
1. The label column should contain "ham" for non-spam messages and "spam" for spam messages. you may add other columns but those would be ignored
2. The optional commands are
   - --batch_size. default is 64
   - --max_epochs. default is 2
3. --mlflow_tracking_uri is optional. if it's ommited, the vocab file and checkpoint file are saved at the root. with mlflow server running and --mlflow_tracking_uri included in the arguments, this will upload the vocab file and checkpoint file to your local mlflow
4. During testing, when i was using distributed processing with mlflow tracking using sqlite, the part which saves artifacts to mlflow kept crashing even if i had everything setup so that only rank 0 will handle saving. it does sometimes work but just in case you're having issues with training using mlflow with sqlite, don't include this argument. You can run distributed training consistently if you remove ```--mlflow_tracking_uri``` while ```--distributed_processing``` is on although this will save the final checkpoint and vocab files at the root. you can try to keep running the training until it works. Note that when turned on, it will use the gpu for training.


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
docker run -it --rm -p 9696:9696 -e RUN_ID=73b284d54f1d46d0a0ac09b058dc33a4 -e TRACKING_URI="http://host.docker.internal:5001" --add-host=host.docker.internal:host-gateway spam-prediction-service-mlflow:v1
```

to test that the container can connect to your mlflow setup, run the following in the container
```
curl http://host.docker.internal:5001
```

check your local mlflow server. it should return a 200 response