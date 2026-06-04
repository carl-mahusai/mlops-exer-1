# mlops-exer-1

Development
1. start mlflow at the root folder like so
```
mlflow server --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

this will create an sqlite file named mlflow.db for the backend store where runs and artifacts are stored and a folder named mlruns which will contain

you can view mlflow in
```
http://127.0.0.1:5001
```