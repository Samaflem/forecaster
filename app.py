# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 23:41:20 2020

@author: Sam
"""

#  %% Libraries
import streamlit as st
import io
import numpy as np  # For Linear Algebra Data Manipulations
import pandas as pd  # I/O Data Manipulation
from statsmodels.tsa.arima.model import ARIMA
import warnings
import base64
import os
import json
import pickle
import uuid
import re
warnings.filterwarnings("ignore")

st.set_option('deprecation.showfileUploaderEncoding', False)
#  %%
st.write("""
# Average Balance Forecast""")

st.write(" # Input Features")

uploaded_file = st.file_uploader("Upload your input excel file",type = ["xlsx"],  encoding = None)

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, sheet_name = None, engine='openpyxl')
else:
    st.write("""
### Please upload a valid excel data file.
              """)

              
frame = pd.DataFrame()
p, d, q = list(range(0, 4)), list(range(0, 3)), list(range(0, 3))


def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    """
    Generates a link to download the given object_to_download.

    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.

    Returns:
    -------
    (str): the anchor tag to download object_to_download

    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')

    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;

            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

    return dl_link


def file_selector(folder_path='output'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


# %% Functions
def data_prep(data):
    df1 = data.transpose()  # Transpose the data
    df1 = df1.asfreq('M')  # Set the frequency of the data to Months
    df1.columns = ["actual_balance"]  # rename the balance variable
    df1.index.names = ["Date"]  # Rename the index(date) variable
    return df1    


def negative_values(data):
    # check for negative values
    check1 = (data < 0).values.any()
    if check1:
        df1 = ["inactive account/ Loan account"]
    else:
        df1 = data
    return df1


def missing_values(data):
    check2 = (data.isnull().sum() > 0).values
    if check2:
        df2 = data.interpolate(method='linear', limit_direction='backward')
        check3 = (df2.isnull().sum() > 0).values
        if check3:
            df3 = ["inactive account/ Loan account"]
        elif not check3:
            df3 = df2
    elif not check2:
        df3 = data
    return df3    

    
def train_test_split(data, perc):
    n = int(len(data) * (1- perc))
    return data[:n], data[n:]


def data_check(data):
    check = (len(data) == 1)
    return check


def hyper_parameters_search(df1, test_value, p, d, q):
    try:
        params = {}
        if not test_value:
            train, test = train_test_split(df1, 0.1)
            for i in p:
                for j in d:
                    for k in q:
                        order = (i, j, k)
                        predictions = []
                        X_train = train.values
                        X_test = test.values
                        history = [x for x in X_train]
                        for m in range(len(X_test)):
                            model = ARIMA(history, order=order)
                            model_fit = model.fit()
                            pred = model_fit.forecast(1)[0]
                            history.append(X_test[m])
                            predictions.append(pred)

                        rmse = np.sqrt(np.mean((predictions - X_test) ** 2))
                        params.update({order:rmse})
                        key_min = min(params.keys(), key=(lambda k: params[k]))
            return key_min
    except:
        out =  ["Too Many Missing Data"]
    return out


def arima_model(data, order):
    try:
        model = ARIMA(data, order=order, enforce_stationarity=False)
        results = model.fit()
        pred = results.forecast(1)[0]
        return pred
    except:
        pass


# %% Processing Key
if st.button("Process"):
    for key in df.keys():
        df1 = pd.DataFrame(df[key])
        df1 = data_prep(df1)
        df2 = df1.copy()
        df2 = negative_values(df2)
        test_value1 = data_check(df2)
        if not test_value1:
            df2 = missing_values(df2)
            test_value2 = data_check(df2)
            if not test_value2:
                order = hyper_parameters_search(df2, test_value = test_value2, p = p, d = d, q = q)
                test_value3 = data_check(order)
                if not test_value3:
                    pred = arima_model(df2, order)
                else:
                    pred = order
            else:
                pred = df2
        else:
            pred = df2
        
        pred1 = pred
        pred = pd.Series(pred)
        n = len(df1)
        t = n-3
        df4 = np.array(df1[t:n].transpose())
        df4 = pd.DataFrame(df4.tolist(), columns=["t-2", "t-1", "t"])
        pred = pd.DataFrame(pred, columns=["Next Month Average Balance"])
        cus_id = pd.Series(key)
        cus_id = pd.DataFrame(cus_id, columns=["Customer ID"])
        df5 = pd.concat([cus_id, df4, pred], axis=1)
        try:
            test = isinstance(float(pred1), (int, float))
            if test:
                df5["% Change"] = round((df5["Next Month Average Balance"] - df5["t"])/df5["t"], 2)
        except Exception:
            df5["% Change"] = df5["Next Month Average Balance"]
        frame = frame.append(df5)

    frame.to_excel("./output/forecast.xlsx", index=False)
    # Upload file for testing
    folder_path = './output'
    filename = file_selector(folder_path=folder_path)

    # Load selected file
    with open(filename, 'rb') as f:
        s = f.read()

    download_button_str = download_button(s, filename, f'Click here to download {filename}')
    st.markdown(download_button_str, unsafe_allow_html=True)

    del cus_id, df, df1, df2, n, t, p, d, q, frame, key, df4, df5
    del order, pred, test_value1, test_value2, test_value3

