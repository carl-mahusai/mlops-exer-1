# mlops-exer-1

Development
1. Create your virtual environment. Note that this was made with Python 3.14.5

1. Open your CLI and run
```
pip install -r requirements/requirements-dev.txt
```

this will install the dev requirements

2. start mlflow at the root folder like so
```
mlflow server --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

this will create an sqlite file named mlflow.db for the backend store where runs and artifacts are stored and a folder named mlruns which will contain your experiment runs

you can view mlflow in
```
http://127.0.0.1:5001
```

3. Make sure you're at the root of the repo then run the following from your CLI to initiate training
```
python -m training.run_training --data=<path to input data> --name_of_label_column=<column where the labels are> --name_of_message_column=<name of column where the messages are> --mlflow_tracking_uri='sqlite:///mlflow.db'
```

for example
```
python -m training.run_training --data='test_dataset/spam.csv' --name_of_label_column='v1' --name_of_message_column='v2' --mlflow_tracking_uri='sqlite:///mlflow.db'
```

you can run with distributed processing by adding the ```--distributed_processing``` argument
```
python -m training.run_training --data='test_dataset/spam.csv' --name_of_label_column='v1' --name_of_message_column='v2' --mlflow_tracking_uri='sqlite:///mlflow.db' --max_epoch=20 --distributed_processing
```

but it may crash when using mllflow with an sqlite db. Notes below

Notes regarding the training script
1. The label column should contain "ham" for non-spam messages and "spam" for spam messages. you may add other columns but those would be ignored
2. The optional commands are
   - --batch_size. default is 64
   - --max_epochs. default is 2
3. --mlflow_tracking_uri is optional. if it's ommited the vocab file and checkpoint file are saved at the root. with mlflow server running and --mlflow_tracking_uri included in the arguments, this will upload the vocab file and checkpoint file to your local mlflow
4. During testing, when i was using distributed processing with mlflow tracking using sqlite, the part which saves artifacts to mlflow kept crashing even if i had everything setup so that only rank 0 will handle saving. it does sometimes work but just in case you're having issues with training using mlflow with sqlite, don't include this argument. Note that when turned on, it will use the gpu for training.


4. Build the docker file for the predictor
stay on the parent directory and run the following docker build command. 
```
docker build -f deployment/Dockerfile -t spam-prediction-service-mlflow:v1 .
``

to run the container
```
docker run -it --rm -p 9696:9696  spam-prediction-service-mlflow:v1
```