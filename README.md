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

this will create an sqlite file named mlflow.db for the backend store where runs and artifacts are stored and a folder named mlruns which will contain

you can view mlflow in
```
http://127.0.0.1:5001
```

3. Run the following from your CLI to initiate training
```
python -m training.run_training --data=<path to input data> --name_of_label_column=<column where the labels are> --name_of_message_column=<name of column where the messages are>
```

Notes regarding the training script
1. The label column should contain "ham" for non-spam messages and "spam" for spam messages. you may add other columns but those would be ignored
2. The optional commands are
   - --batch_size. default is 64
   - --max_epochs. default is 2




