import mlflow
import torch
import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import ClassificationPreset
from evidently.core.datasets import BinaryClassification
from pathlib import Path
from lightning.pytorch.utilities.rank_zero import rank_zero_only

@rank_zero_only
def generate_evidently_report(dm, model, mlflow_logger, args):

    predictions = []
    targets = []
    scores = []

    model.eval()

    device = model.device

    for batch in dm.test_dataloader():

        x, y = batch

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():

            logits = model(x)

            probs = torch.sigmoid(logits)

            preds = (probs > 0.5).float()

        predictions.extend(
            preds.cpu().numpy().astype(int)
        )

        targets.extend(
            y.cpu().numpy().astype(int)
        )

        # scores.extend(
        #     probs.cpu().numpy()
        # )

        scores.extend(
            probs.cpu().numpy().flatten().tolist()
        )

    eval_df = pd.DataFrame({
        "target": targets,
        "prediction": predictions,
        "score": scores
    })

    # dataset = Dataset.from_pandas(
    #     eval_df,
    #     data_definition=DataDefinition(
    #         classification={
    #             "target": "target",
    #             "prediction_labels": "prediction"
    #         }
    #     )
    # )

    dataset = Dataset.from_pandas(
        eval_df,
        data_definition=DataDefinition(
            classification=[
                BinaryClassification(
                    target="target",
                    prediction_labels="prediction",
                    prediction_probas="score"
                )
            ]
        )
    )

    report = Report(
        metrics=[ClassificationPreset()]
    )

    script_dir = Path(__file__).resolve().parent.parent
    reports_dir = script_dir / "reports"

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(exist_ok=True)

    report_path = reports_dir / "classification_report.html"

    snapshot = report.run(dataset)

    snapshot.save_html(str(report_path))

    if mlflow_logger:
        with mlflow.start_run(run_id=mlflow_logger.run_id):
            mlflow.log_artifact(
                str(report_path),
                artifact_path="reports"
            )

            # mlflow.log_artifact(
            #     str(json_path),
            #     artifact_path="reports"
            # )