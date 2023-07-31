import os
import argparse
import pickle

import mlflow
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import xgboost as xgb

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("news-classification-experiment")

mlflow.xgboost.autolog()

@task(log_prints=True)
def load_pickle(filename):
	with open(filename, "rb") as f_in:
		return pickle.load(f_in)

@task(log_prints=True)
def objective(params):
	with mlflow.start_run():
		mlflow.set_tag("model", "xgboost")
		mlflow.log_params(params)

		model = xgb.XGBClassifier(**params)
		model = model.fit(X_train, y_train)

		# Predicting values for training and validation data, and getting prediction probabilities
		y_pred = model.predict(X_val)
		accuracy = accuracy_score(y_val, y_pred)

		print("SCORE:", accuracy)

		mlflow.log_metric("accuracy", accuracy)


	return {'loss': accuracy, 'status': STATUS_OK}

@flow(log_prints=True)
def run_optimization(data_path: str, num_trials: int = 5):
	global X_train, y_train, X_val, y_val
	X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
	X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

	search_space = {
	'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
	'eval_metric': 'auc',
	'learning_rate': hp.loguniform('learning_rate', -3, 0),
	'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
	'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
	'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
	'objective': 'binary:logistic',
	'seed': 42
	}

	best_result = fmin(
		fn=objective,
		space=search_space,
		algo=tpe.suggest,
		max_evals=num_trials,
		trials=Trials()
	)
	# Querying mlflow api instead of using web UI.
	# Sorting by validation aucroc and then getting top run for best run.
	EXPERIMENT_ID = mlflow.get_experiment_by_name('news-classification-experiment').experiment_id
	runs_df = mlflow.search_runs(experiment_ids=EXPERIMENT_ID, order_by=['metrics.accuracy DESC'])
	best_run = runs_df.iloc[0]
	best_run_id = best_run['run_id']

	# Loading model from best run
	best_model = mlflow.xgboost.load_model('runs:/' + best_run_id + '/model')
	y_predict_model = best_model.predict(X_val)

	# Predicting and evaluating best model on holdout set'

	model_report = classification_report(y_val, y_predict_model)
	print(model_report)

	# Plotting the area under the curve to visualize the performance of the model

	pred_prob = best_model.predict_proba(X_val)[:, 1]

	auc_score = roc_auc_score(y_val, pred_prob)
	print("AUC_SCORE:", auc_score)

	markdown__accuracy_report = f""" News Classification Model Report

		{model_report}


		AUC_SCORE: {auc_score}
		"""

	create_markdown_artifact(
		key="news-classifier-report", markdown=markdown__accuracy_report
	)

	mlflow.xgboost.log_model(best_model, artifact_path="\model")
	mlflow.register_model(f'runs:/{best_run_id}/artifacts/model', 'NewsClassification-XGBHP')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Training Pipeline')
	parser.add_argument('--data_path', required=True, help='Destination of processed data')
	args = parser.parse_args()
	run_optimization(args.data_path)
