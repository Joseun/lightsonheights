# Project Title
Source Based Fake News Classification.

# Project Description
Social media is a vast pool of content, and among all the content available for users to access, news is an element that is accessed most frequently. This news can be posted by politicians, news channels, newspaper websites, or even common civilians. These posts must be checked for their authenticity, since spreading misinformation has been a real concern in today’s times, and many firms are taking steps to make the common people aware of the consequences of spreading misinformation. The measure of authenticity of the news posted online cannot be definitively measured, since the manual classification of news is tedious and time-consuming and is also subject to bias.

## Project

This repository contains a Python Streamlit application for analyzing news data, as well as Jupyter Notebooks for building the model. It also contains with python pipelines for preprocessing, training, monitoring model inferences 

## Built With

* [Python](https://www.python.org/) - The programming language used
* [Streamlit](https://streamlit.io/) - The framework used 
* [Mlflow](https://mlflow.org/) - For experiment tracking
* [Prefect](https://www.prefect.io/opensource/) - For workflow orchestration
* [Evidently](https://www.evidentlyai.com/) - For model monitoring
## Deployment

Click here to get to the deployed [News Post Checker Application](https://news-post-checker.onrender.com/)

## Authors

* **Joseph Ologunja** - *Initial work* - [Joseun](https://github.com/joseun)

## Repository Structure
| Folder/Code | Content |
| ------------- | ------------- |
| .streamlit | Contains the config.toml to set certain design parameters |
| Train data | Contains the data used in training the model CSV format |
| Test data | Contains the data used in test the model excel format |
| Submission | Contains the labelled test data using the model in excel format |
| News_Classfication.ipynb | Contains the code for data exploration, analysis, visualization and model building |
| app.py | Contains the actual Streamlit application |
| model | Contains the trained model in pickled format |
| tokenizer | Contains the tokenizer in pickled format |
| requirements.txt | Contains all requirements (necessary for Streamlit deployment) |
