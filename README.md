# mlops-exer-1

**Development**

0. If like me and you're running everything in WSL, have to increase the RAM available to WSL otherwise it will crash. My wslconfig looks like this
```
[wsl2]
memory=8GB   # Limits VM memory in WSL
processors=4 # Makes the WSL 4 VM use two virtual processors
swap=4GB
kernelCommandLine=systemd.unified_cgroup_hierarchy=1

[experimental]
autoMemoryReclaim=gradual
```
This is on 16GB on RAM

Note also that i included sample datasets in the test_dataset folder. The ```spam.csv``` file is the full dataset from here https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset. ```spam_sample.csv``` is a smaller dataset for quick testing to make sure that things work if within memory limits

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

if you want to use postgresql for the backend-store-uri and you're using WSL2 and windows, do the following
1. Install postgresql
2. Open ```Windows Defender Firewall with Advanced Security```
3. Click ```New Rule...```
4. Select ```Port``` for rule type
5. Select ```TCP``` and for ```Specific local ports``` enter ```5432```
6. Select ```Allow the connection```. Connecting from WSL2 won't be secure so don't select the secure option
7. Select at least ```Public```. Can select ```Domain``` and ```Private``` as well. I could only connect if ```Public``` was selected
8. Name the rule e.g. ```Postgres - connect from WSL2``` and create it
9. Right click newly created rule and select ```Properties``` then click on the ```Scope``` tab
10. Under ```Remote IP address```, select ```These IP addresses``` then click ```Add...``` and enter range ```172.0.0.1``` to ```172.254.254.254```
11. Repeat step 9 for IP address range ```192.0.0.1``` to ```192.254.254.254```
12. Click ```Apply``` then ```OK```
13. Make sure rule is enabled

Afterwards, configure postgres to accept WSL2 connections

1. Go to ```C:\Program Files\PostgreSQL\$VERSION\data```
2. Verify that postgresql.conf has following set:
```
listen_addresses = '*'
```
3. Update pg_hba.conf to allow connections from WSL2 range e.g. for Postgresl 12:
```
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# IPv4 local connections:
host    all             all             127.0.0.1/32            md5
host    all             all             172.0.0.0/8             md5
host    all             all             192.0.0.0/8             md5
```

For Postgresql 13+ you should use scram-sha-256
4. Restart postgresql

In your bash loading file, like ```.bashrc```, ```.zshrc```, or ```.bash_profile```, add the following
```
# Map Windows host IP as winhost (based on default route, not /etc/resolv.conf)
if ! grep -q 'winhost' /etc/hosts; then
  echo 'Adding Windows host IP as winhost...'
  WINIP=$(ip route | grep default | awk '{print $3}')
  echo -e "\n# Windows host for PostgreSQL\n$WINIP   winhost" | sudo tee -a /etc/hosts
fi
```
this adds the host ip address to your host and assigns it to ```winhost```.

5. Install PSQL only in WSL2. the postgresql server is already running in the host and you don't want an extra postgresql server running in WSL2
6. try logging in to postgresql
```
psql -h winhost -p 5432 -U postgres
```

once you can connect to your postgresql server in the host from WSL2, do the following
a. install ```psycopg2-binary``` in the virtual env of the exercise. this is already in the requirements file
```
pip install psycopg2-binary
```
b. create a new mlflow db, let's say ```mlflow_metrics```. also create a new user in postgresql, let's also call it ```mlflow_metrics```
c. change the owner of your mlflow db to your mlflow user and update the privileges.
```
GRANT ALL PRIVILEGES ON DATABASE mlflow_metrics TO mlflow_metrics;
ALTER DATABASE mlflow_metrics OWNER TO mlflow_metrics;
```
d. Try logging in to your mlflow metrics db with the mlflow metrics user


Once that's done, you can change your mlflow server's backend-store-uri like this
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

training is set up for distributed processing via deepspeed. you would have to install the nvidia cuda tooklit to run the trainer
```
sudo apt update
sudo apt install nvidia-cuda-toolkit
```


distributed processing works best when ```--accelerator``` is set to ```gpu```. also set the number of devices using ```--devices```

```
python -m training.run_training --data='test_dataset/spam.csv' --name_of_label_column='v1' --name_of_message_column='v2' --mlflow_tracking_uri='http://127.0.0.1:5001' --max_epoch=5 --accelerator="gpu" --devices=1
```

I tried using ddp for the training but that was causing very frequent crashes. ```fsdp``` sometimes crashes as well but not as frequently as ```ddp```. ```deepspeed``` was a very consistent performer which is why i used it.

the training script also has a hyperparameter optimization setup. it has two modes. the first one is triggered by calling the ```training.run_optimization``` module. the hyperparameter optimization returns a dictionary of the hyperparameters for training and data module setup. something like this

```
python -m training.run_optimization --data='test_dataset/spam.csv' --name_of_label_column='v1' --name_of_message_column='v2' --mlflow_tracking_uri='http://127.0.0.1:5001' --max_epoch=20 --accelerator="gpu" --devices=1
```

it would print something like this
```
{
  'batch_size': 16, 
  'embedding_dim': 32, 
  'hidden_dim': 128, 
  'lr': 0.0034370889603316875, 
  'max_vocab_size': 5000, 
  'max_length': 82
}
```

or
```
{
  'batch_size': 8, 
  'embedding_dim': 256, 
  'hidden_dim': 128, 
  'lr': 0.0001805834262638685, 
  'max_vocab_size': 5000, 
  'max_length': 58
}
```
you can then call training with these hyperparameters like this

```
python -m training.run_training --data='test_dataset/spam.csv' --name_of_label_column='v1' --name_of_message_column='v2' --mlflow_tracking_uri='http://127.0.0.1:5001' --max_epoch=20 --accelerator="gpu" --devices=1 --batch_size=16 --embedding_dim=32 --hidden_dim=128 --lr=0.0034370889603316875 --max_vocab_size=5000 --max_length=82
```

or
```
python -m training.run_training --data='test_dataset/spam.csv' --name_of_label_column='v1' --name_of_message_column='v2' --mlflow_tracking_uri='http://127.0.0.1:5001' --max_epoch=20 --accelerator="gpu" --devices=1 --batch_size=8 --embedding_dim=256 --hidden_dim=128 --lr=0.0001805834262638685 --max_vocab_size=5000 --max_length=58
```

the other option is calling the ```training.run_training``` module and adding the ```--optimize_and_train``` parameter. this will run hyperparameter tuning and right after, run the final training with the optimized hyperparameters

```
python -m training.run_training --data='test_dataset/spam.csv' --name_of_label_column='v1' --name_of_message_column='v2' --mlflow_tracking_uri='http://127.0.0.1:5001' --max_epoch=20 --accelerator="gpu" --devices=1 --optimize_and_train
```


Notes regarding the training script
1. The label column should contain "ham" for non-spam messages and "spam" for spam messages. you may add other columns aside from the label and messages column but those would be ignored
2. The optional commands are
   - --batch_size - default is 8
   - --max_epoch - default is 2
   - --num_nodes - number of separate machines to train on. default is 1
   - --devices - devices to use in each training node. default is "1"
3. For the --batch_size argument, training uses gradient accumulation with a value of 4. so the effective batch size is ```--batch_size x 4```. So for the default value of 8, the effective batch size memory would be 8 while running a batch size of 32.
3. Training has early stopping in place where it will stop after 3 validation checks with no decrease in training loss
3. --mlflow_tracking_uri is optional. if it's ommited, the vocab file and checkpoint file are saved at the root. with mlflow server running and --mlflow_tracking_uri included in the arguments, this will upload the vocab file and checkpoint file to your local mlflow
4. If you're on a laptop which uses nvidia optimus, make sure to set it to use your discrete gpu. I experienced frequent crashes because I believe it was using integrated graphics. If you still experience crashes, don't use distributed processing and test the training script


5. Build the docker file for the predictor
stay on the parent directory and run the following docker build command. 
```
docker build -f deployment/endpoint/Dockerfile -t spam-prediction-service-mlflow:v1 .
```

to run the container locally
```
docker run -it --rm -p 9696:9696 -e RUN_ID=<run id in mlflow> -e TRACKING_URI=<tracking uri used by mlflow> spam-prediction-service-mlflow:v1
```

for a local run, add ```--add-host=host.docker.internal:host-gateway``` and use "http://host.docker.internal:5001" for the tracking uri. this will connect to the mlflow setup running 
```
docker run -it --rm -p 9696:9696 -e RUN_ID=a29d61564ccc4565adb883d02fb5ef9b -e TRACKING_URI="http://host.docker.internal:5001" --add-host=host.docker.internal:host-gateway spam-prediction-service-mlflow:v1
```

to test that the blackbox container running locally can connect to your mlflow setup that's also running locally, run the following in the container
```
curl http://host.docker.internal:5001
```

check your local mlflow server. it should return a 200 response

to test the docker container, from the root folder, run this in your CLI
```
python deployment/endpoint/test.py
```

to launch the sample gradio interface, from the root folder run this in your CLI
```
python deployment/gradio/interface.py --api-url=<prediction endpoint>
```

for the local example, this would be
```
python deployment/gradio/interface.py --api-url=http://localhost:9696/predict
```

**Running in Prefect**

1. setup the postgresql db to be used by prefect. setup the connection string to be used
```
postgresql+asyncpg://<username>:<password>@<host>:<port>/<database_name>
```

for my local setup, it would be like this
```
postgresql+asyncpg://prefect:vGJJx0Su@winhost:5432/prefect
```

2. set the api connection url. either via an exported variable like this
```
export PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://prefect:vGJJx0Su@winhost:5432/prefect"
```
or globally like this
```
prefect config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://prefect:vGJJx0Su@winhost:5432/prefect"
```
Note that you can use both. the default that will be used is the config but if you do the export, that will be the string used

3. update also the privileges of the user for the db
```
GRANT ALL PRIVILEGES ON DATABASE <database_name> TO <username>;
GRANT USAGE, CREATE ON SCHEMA public TO <username>;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO <username>;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO <username>;
ALTER DATABASE <database_name> OWNER TO <username>;
```

then run
```
prefect server database upgrade
```

3. once that's done, run
```
prefect server start
```
you only have to setup the db once. you can just run prefect server start in one window for future runs

4. in another window, get the prefect api url and set it like this
```
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

5. Run this to create your new process pool
```
prefect work-pool create my-local-pool --type process
```

this creates a process type prefect pool to be used by the deployment

6. start this pool
```
prefect worker start --pool my-local-pool
```
no need to recreate the pool for future runs.

7. run this in a third window
```
prefect deploy
```

this will read the prefect.yaml file and setup the deployment
select the deployment you want setup, it should look something like this
```
┏━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃    ┃ Name          ┃ Entrypoint                                                     ┃ Description                    ┃
┡━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ >  │ spam-training │ orchestration/pipelines/training_pipeline.py:training_pipeline │ None                           │
│    │               │                                                                │ No, configure a new deployment │
└────┴───────────────┴────────────────────────────────────────────────────────────────┴────────────────────────────────┘
```

I selected spam-training

prefect will ask if you want prefect to pull code. since this is a local run, select ```n```
```
? Your Prefect workers will need access to this flow's code in order to run it. Would you like your workers to pull your
flow code from a remote storage location when running this flow? [y/n] (y): n
```
it should display something like this
```
Your Prefect workers will attempt to load your flow from:
/mnt/c/Users/user/Documents/Projects/mlops-exer-1/orchestration/pipelines/training_pipeline.py. To see more options for
managing your flow's code, run:

        $ prefect init
```

as you can see, it's pointing to the current folder
it will then ask this

```
Would you like to configure schedules for this deployment? [y/n] (y): n
```
since this is a local run and i can deploy anytime, select ```n```

8. Run the following in your terminal for the worker
```
prefect config set PREFECT_API_REQUEST_TIMEOUT=300
prefect config set PREFECT_WORKER_HEARTBEAT_SECONDS=60
```

9. once finished, in that same window, run the deployment. this example which uses ```spam_sample.csv``` uses a small sample for testing and would run the optimize section before training

```
prefect deployment run 'training-pipeline/spam-training' \
    --param data='/mnt/c/Users/user/Documents/Projects/mlops-exer-1/test_dataset/spam_sample.csv' \
    --param name_of_label_column='v1' \
    --param name_of_message_column='v2' \
    --param mlflow_tracking_uri='http://127.0.0.1:5001' \
    --param max_epoch=20 \
    --param accelerator='gpu' \
    --param devices=1 \
    --param optimize_and_train=true \
    --param n_trials=5
```

run this for the full dataset
```
prefect deployment run 'training-pipeline/spam-training' \
    --param data='/mnt/c/Users/user/Documents/Projects/mlops-exer-1/test_dataset/spam.csv' \
    --param name_of_label_column='v1' \
    --param name_of_message_column='v2' \
    --param mlflow_tracking_uri='http://127.0.0.1:5001' \
    --param max_epoch=20 \
    --param accelerator='gpu' \
    --param devices=1 \
    --param optimize_and_train=true \
    --param n_trials=5
```

it runs the exact same code as the command line version
```
python -m training.run_training --data='test_dataset/spam.csv' --name_of_label_column='v1' --name_of_message_column='v2' --mlflow_tracking_uri='http://127.0.0.1:5001' --max_epoch=20 --accelerator="gpu" --devices=1 --optimize_and_train --n_trials=5
```

