#!/usr/bin/python3
""" NEWS POST CLASSIFIER MODULE"""
import time
import pickle
import warnings
from datetime import date

import pandas as pd
import requests
import streamlit as st
from xgboost import XGBClassifier
from streamlit_lottie import st_lottie, st_lottie_spinner

warnings.filterwarnings("ignore")

st.set_page_config(page_title="News Post Checker App", layout="wide")


def load_lottieurl(url: str):
    """Sends a request for gifs and icons

    Args:
        url: string of url for lottie content

    Return:
        object: json
    """
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_book = load_lottieurl(
    "https://assets9.lottiefiles.com/packages/lf20_zn4imzfv.json"
)
st_lottie(lottie_book, speed=1, height=200, key="initial")

# loading in the model to predict on the data
model = XGBClassifier()

model.load_model("model.json")
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


# defining the function which will make the prediction using
# the data which the user inputs


def prediction(text: str) -> int:
    """Classifies text to 1 or 0

    Args:
        text: text from user

    Return:
        integer: 1 or 0
    """
    X = tokenizer.transform([text]).toarray()
    # st.write(tokenizer.get_feature_names_out())
    X = pd.DataFrame(X, columns=tokenizer.get_feature_names_out())
    prediction = model.predict(X)
    # print(prediction)
    return prediction


# this is the main function in which we define our webpage


def main():
    # here we define some of the front end elements of the web page like
    # the font and background color, the padding and the text to be displayed

    html_temp = f"""
    <div>
        <h1 style ="color:white;text-align:center;">
            Source Based Fake News Classification </h1>
    </div>
    """

    # this line allows us to display the front end aspects we have
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown(
        """
        <div>
        <h3 style ="color:white;text-align:left;">
        Social media is a vast pool of content, and among all the content available for users to access, news is an element that is accessed most frequently. This news \
        can be posted by politicians, news channels, newspaper websites, or even common civilians. These posts must be checked for their authenticity, since \
        spreading misinformation has been a real concern in today’s times, and many firms are taking steps to make the common people aware of the consequences of \
        spreading misinformation.</h3>
    </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """ 
        <div>
        <h4 style ="color:white; text-align:justify;">
        Check the authenticity of a news post. Just fill in the details below👇
        </h4>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # the following lines create widgets in which the user can enter
    # the data required to make the prediction
    author = st.text_input("Author", value="", placeholder="Enter Author here")
    published = st.date_input('Date Published', date.today())
    title = st.text_input("Title", value="", placeholder="Enter title here")
    text = st.text_area("Text", value="", placeholder="Enter news text here")
    language = st.selectbox(
        'Language', ('English', 'Ignore', 'German', 'French', 'Others')
    )
    site_url = st.text_input("Site Url", value="", placeholder="Enter site url here")
    main_img_url = st.text_input(
        "Main Image Url",
        value="",
        placeholder="Enter the url of the main image on the site here",
    )
    genre = st.selectbox(
        'Type',
        (
            'Bias',
            'Conspiracy',
            'Fake',
            'BS',
            'Satire',
            'Hate',
            'Junksci',
            'State',
            'Others',
        ),
    )
    text = author + " " + title + " " + text + " " + site_url + " " + main_img_url

    # the below line ensures that when the button called 'Predict' is clicked,
    # the prediction function defined above is called to make the prediction
    # and store it in the variable result
    result = ""
    lottie_load = load_lottieurl(
        "https://assets5.lottiefiles.com/private_files/lf30_fup2uejx.json"
    )
    if st.button("Check"):
        with st_lottie_spinner(lottie_load, speed=1, height=100, key="load"):
            time.sleep(2)
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
