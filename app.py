#!/usr/bin/python3
""" NEWS POST CLASSIFIER MODULE"""
import pickle
import requests
import streamlit as st
import time
import warnings

from datetime import date
from streamlit_lottie import st_lottie, st_lottie_spinner

warnings.filterwarnings("ignore")

st.set_page_config(page_title="News Post Checker App", layout="wide")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_book = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_zn4imzfv.json")
st_lottie(lottie_book, speed=1, height=200, key="initial")
# loading in the model to predict on the data
with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)


def welcome():
    return 'welcome all'

# defining the function which will make the prediction using
# the data which the user inputs


def prediction(text):
    X = tokenizer.transform([text]).toarray()
    prediction = model.predict(X)
    # print(prediction)
    return prediction

# this is the main function in which we define our webpage


def main():
    # giving the webpage a title
    # st.title("News Post Checker")

    # here we define some of the front end elements of the web page like
    # the font and background color, the padding and the text to be displayed
    html_temp = f"""
    <style>
        .stApp {{
            background-image: url("https://images.pexels.com/photos/743986/pexels-photo-743986.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
            background-attachment: auto;
            background-size: cover
        }}
    </style>
    <div>
        <h1 style ="color:black;text-align:center;">
            Lights on Heights News Post Checker </h1>
    </div>
    """

    # this line allows us to display the front end aspects we have
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html=True)

    # the following lines create text boxes in which the user can enter
    # the data required to make the prediction
    author = st.text_input("Author", value="", placeholder="Enter Author here")
    published = st.date_input('Date Published', date.today())
    title = st.text_input("Title", value="", placeholder="Enter title here")
    text = st.text_area("Text", value="", placeholder="Enter news text here")
    language = option = st.selectbox('Language', ('English', 'Ignore', 'German', 'French', 'Others'))
    site_url = st.text_input("Site Url", value="", placeholder="Enter site url here")
    main_img_url = st.text_input("Main Image Url", value="", placeholder="Enter the url of the main image on the site here")
    genre = option = st.selectbox('Type', ('Bias', 'Conspiracy', 'Fake', 'BS', 'Satire', 'Hate', 'Junksci', 'State', 'Others'))
    text = author + " " + title + " " + text + " " + site_url + " " + main_img_url
    # st.write(text)
    result = ""
    # the below line ensures that when the button called 'Predict' is clicked,
    # the prediction function defined above is called to make the prediction
    # and store it in the variable result
    lottie_load = load_lottieurl("https://assets5.lottiefiles.com/private_files/lf30_fup2uejx.json")
    if st.button("Check"):
        with st_lottie_spinner(lottie_load, speed=1, height=100, key="load"):
            time.sleep(5)
        if text == "    ":
            st.warning('No news post entered', icon="⚠️")
        else:
            result = prediction(text)
            if result == 1:
                result = 'credible'
                st.success(f"This news post seems {result}", icon="✅")
            else:
                result = 'inaccurate'
                st.error(f"This news post seems {result}", icon="🚨")

if __name__ == '__main__':
    main()
