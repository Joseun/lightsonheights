import os
import argparse

import nltk
import mlflow
import pandas as pd
from prefect import flow, task
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    TextDescriptorsDriftMetric
)
from prefect.artifacts import create_markdown_artifact

from training import load_pickle

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("news-classification-experiment")


nltk.download('words')

column_mapping = ColumnMapping(
    target='label',
    prediction='label',
    text_features=['corpus'],
)

report = Report(
    metrics=[
        ColumnDriftMetric(column_name='label'),
        TextDescriptorsDriftMetric(column_name='corpus'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ]
)


@task(log_prints=True)
def read_dataframe(filename: str):
    df = pd.read_excel(filename)
    return df


@task(log_prints=True)
def calculate_metrics(reference_data_path: str, current_data):

    raw_data = pd.read_csv(reference_data_path)
    reference_data = pd.DataFrame()
    reference_data['corpus'] = (
        raw_data['author']
        + ' '
        + raw_data['title']
        + ' '
        + raw_data['text']
        + ' '
        + raw_data['language']
        + ' '
        + raw_data['site_url']
        + ' '
        + raw_data['main_img_url']
        + ' '
        + raw_data['type']
    )

    reference_data['label'] = raw_data['label']

    reference_data.dropna(inplace=True)

    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']

    drift_report = f""" Model Data Drift Report

		predicted drift: {prediction_drift}
		number of drifted columns: {num_drifted_columns}
		"""
    print(drift_report)
    create_markdown_artifact(
        key="news-classifier-prediction-report", markdown=drift_report
    )


@task(log_prints=True)
def test_model(X_test):
    EXPERIMENT_ID = mlflow.get_experiment_by_name(
        'news-classification-experiment'
    ).experiment_id
    runs_df = mlflow.search_runs(
        experiment_ids=EXPERIMENT_ID, order_by=['metrics.accuracy DESC']
    )
    best_run = runs_df.iloc[0]
    best_run_id = best_run['run_id']

    # Loading model from best run
    model = mlflow.xgboost.load_model(f'runs:/{best_run_id}/model')
    y_pred = model.predict(X_test)
    return y_pred


@flow(log_prints=True)
def batch_monitoring(test_data_path: str, train_data_path: str, dest_path: str):
    df = read_dataframe(test_data_path)
    test_data = pd.DataFrame()
    test_data['corpus'] = (
        df['author']
        + ' '
        + df['title']
        + ' '
        + df['text']
        + ' '
        + df['language']
        + ' '
        + df['site_url']
        + ' '
        + df['main_img_url']
        + ' '
        + df['type']
    )

    vectorizer = load_pickle(os.path.join(dest_path, "vectorizer.pkl"))
    test_data = test_data.dropna()
    X_test = vectorizer.transform(test_data.corpus).toarray()

    X_test_df = pd.DataFrame(X_test, columns=vectorizer.get_feature_names_out())
    y_test = test_model(X_test=X_test_df)
    test_data['label'] = y_test

    label_dict = {'1': 'Real', '0': 'Fake'}
    test_data['label'] = test_data['label'].astype(str).map(label_dict)
    calculate_metrics(train_data_path, test_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction Monitoring Pipeline')
    parser.add_argument('--test_data_path', required=True, help='Location of test data')
    parser.add_argument(
        '--train_data_path', required=True, help='Location of train data'
    )
    parser.add_argument('--dest_path', required=True, help='Location of processed data')
    args = parser.parse_args()
    batch_monitoring(args.test_data_path, args.train_data_path, args.dest_path)
