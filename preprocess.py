import os
import argparse
import pickle
import pandas as pd

import nltk
import re

from prefect import flow, task
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings

nltk.download('stopwords')

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("news-classification-experiment")

@task(log_prints=True)
def dump_pickle(obj, filename: str):
	with open(filename, "wb") as f_out:
		return pickle.dump(obj, f_out)

@task(log_prints=True)
def read_dataframe(filename: str):
	df = pd.read_csv(filename)
	return df

@task(log_prints=True)
def clean_dataframe(df: pd.DataFrame):
	cleaned_data = df.dropna()
	print(cleaned_data.isnull().sum())
	cleaned_data = cleaned_data.convert_dtypes()
	print(cleaned_data.info())
	print(cleaned_data.shape)
	cleaned_data = cleaned_data.drop_duplicates()
	print(cleaned_data.shape)
	return cleaned_data

@flow(name="Subflow", log_prints=True)
def clean_text_data(df):
	text_data = pd.DataFrame()
	text_data['corpus'] = df['author'] + ' ' +  df['title'] + ' ' + \
						  df['text'] + ' ' + df['language'] + ' ' + \
						  df['site_url'] + ' ' + \
						  df['main_img_url'] + ' ' + df['type']
	text_data['corpus'] = text_data['corpus'].str.lower()
	stop = stopwords.words('english')
	text_data['corpus'] = text_data['corpus'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
	search_string = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
	text_data['corpus'] = text_data['corpus'].str.replace(search_string, '')
	text_data['corpus'] = text_data['corpus'].str.replace(r'[\'-]', '')
	text_data['corpus'] = text_data['corpus'].str.replace(r'[\'\",()*&^%$#@!~`+=|/<>?{}\[\]\/\\:;\_]]', ' ')
	text_data['corpus'] = text_data['corpus'].apply(lambda x: ' '.join([word for word in x.split() if not word.startswith('url')]))
	text_data['corpus'] = text_data['corpus'].apply(lambda x: ' '.join([word for word in x.split(' ') if len(word)<25]))
	text_data['corpus'] = text_data['corpus'].map(lambda x: re.sub(r'\W+', ' ', x))
	text_data['corpus'] = text_data['corpus'].str.replace(r'[0-9]', '')
	text_data['corpus'] = text_data['corpus'].str.replace(r'[^a-z]', ' ')

	def lemmatize_text(text: str):
		lemmatizer = WordNetLemmatizer()
		return [lemmatizer.lemmatize(w) for w in word_tokenize(text)]

	# Further text processing with lemmatizing
	text_data['corpus'] = text_data['corpus'].apply(lemmatize_text)
	text_data['corpus'] = [' '.join(map(str, l)) for l in text_data['corpus']]
	text_data['label'] = df['label']
	return text_data


@task(log_prints=True)
def preprocess(df: pd.DataFrame):
	label_dict = { 'Real': '1' , 'Fake': '0' }
	df['label'] = df['label'].map(label_dict).astype(int)

	# The corpus of text will be converted to vectors
	vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

	# The vectorized train corpus is then transformed and placed in a variable
	X = vectorizer.fit_transform(df['corpus']).toarray()   # independent
	y = df['label']   # dependent

	# The variable is placed in a dataframe for ease of processing
	X = pd.DataFrame(X, columns=vectorizer.get_feature_names_out())

	return X, y, vectorizer

@flow(log_prints=True)
def run_data_prep(raw_data_path: str, dest_path: str):
	# Load csv files
	df_raw = read_dataframe(args.raw_data_path)

	# Clean the dataframe
	df_train = clean_dataframe(df_raw)

	model_data = clean_text_data(df_train)

	# Fit the DictVectorizer and preprocess data
	X, y, vectorizer = preprocess(model_data)

	# The train data is spilt into training and validation set at a ratio of 80:20
	X_train, X_val, y_train, y_val = train_test_split(X, y,
													  test_size=0.2,
													  random_state=15,
													  stratify=y
													)

	# Create dest_path folder unless it already exists
	os.makedirs(args.dest_path, exist_ok=True)

	# Save Vectorizer and datasets
	dump_pickle(vectorizer, os.path.join(args.dest_path, "vectorizer.pkl"))
	dump_pickle((X_train, y_train), os.path.join(args.dest_path, "train.pkl"))
	dump_pickle((X_val, y_val), os.path.join(args.dest_path, "val.pkl"))

	mlflow.log_artifact(os.path.join(args.dest_path, "vectorizer.pkl"), artifact_path="vectorizer")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Preprocess Pipeline')
	parser.add_argument('--raw_data_path', required=True, help='Location of raw data')
	parser.add_argument('--dest_path', required=True, help='Destination of processed data')
	args = parser.parse_args()

	run_data_prep(args.raw_data_path, args.dest_path)
