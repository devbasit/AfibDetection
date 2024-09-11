import streamlit as st
st. set_page_config(layout="wide")

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np 
import joblib
from scipy.io import loadmat
from biosppy.signals import ecg
from scipy import stats
from collections import Counter



sigAfib = np.load('./signalAfib.npy', allow_pickle=True)
sigNoSR = np.load('./signalNormal.npy', allow_pickle=True)
sigArrh = np.load('./signalOthers.npy', allow_pickle=True)

healthyDise = joblib.load('NORMAL_DISEASE_DEPLOY.joblib')
AfibOthers  = joblib.load('AFOTHERS_DEPLOY.joblib')


signals = {"AFIB DATA":sigAfib,"NSR Data":sigNoSR, "OTHERS":sigArrh}
extractedFeatures = []
window_size = 10

# plt.style.use("ggplot")
plt.axis('off')
plt.grid(visible = False)

classes = {0: "SINUS RHYTHM", 1:"AFib", 2:'OTHER ARRHYTHMIAS'}

def predict(data):
    array1 = data[:,np.array([5, 6, 1, 4])]
    array2 = data[:,np.array([1,3,4,5,6])]
    pred1  = AfibOthers.predict(array2)
    pred2  = healthyDise.predict(array1)
    for n in range(len(pred2)):
        if pred2[n] == 1:pred2[n] = pred1[n]+1
    counts = Counter(pred2)
    most_occurring_value = stats.mode(pred2.flatten())[0]
    return most_occurring_value, counts[most_occurring_value]/len(pred2)

def extractFeatures(intervals, fs = 500):
        rr = np.array([intervals])/fs
        hr = 60 / rr
        rmssd = np.sqrt(np.mean(np.square(rr)))
        sdnn = np.std(rr)
        mean_rr = np.mean(rr)
        mean_hr = np.mean(hr)
        std_hr = np.std(hr)
        min_hr = np.min(hr)
        max_hr = np.max(hr)
        return [rmssd, sdnn, mean_rr, mean_hr, std_hr, min_hr, max_hr]

def plotEcg(ts=None,
    raw=None,
    filtered=None,
    rpeaks=None,
    heart_rate_ts=None,
    heart_rate=None,
    column = None):

    fig_raw, axs_raw = plt.subplots(4, 1, sharex=True)
    fig_raw.suptitle("ECG Summary")

    # raw signal plot (1)
    axs_raw[0].plot(ts, raw,  label="Raw", color="C0")
    axs_raw[0].set_ylabel("Amplitude")
    axs_raw[0].legend()
    axs_raw[0].grid()

    # filtered signal with R-Peaks (2)
    axs_raw[1].plot(ts, filtered, label="Filtered", color="C0")
    axs_raw[0].set_ylabel("Amplitude")
    axs_raw[0].legend()
    axs_raw[0].grid()

    axs_raw[2].plot(ts, filtered, label="Filtered", color="C0")

    ymin = np.min(filtered)
    ymax = np.max(filtered)
    alpha = 0.1 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    # adding the R-Peaks
    axs_raw[2].vlines(
        ts[rpeaks], ymin, ymax, color="m", label="R-peaks"
    )

    axs_raw[2].set_ylabel("Amplitude")
    axs_raw[2].legend(loc="upper right")
    axs_raw[2].grid()


    # heart rate (3)
    axs_raw[3].plot(heart_rate_ts, heart_rate, label="Heart Rate")
    axs_raw[3].set_xlabel("Time (s)")
    axs_raw[3].set_ylabel("Heart Rate (bpm)")
    axs_raw[3].legend()
    axs_raw[3].grid()

    column.pyplot(fig_raw)


html_temp = """
<h3>
Project Title: The Development of an ECG Analyser and AFib Detector
</h3>

<h4>
Project Overview:
</h4>

<p>
Welcome to our innovative project aimed at revolutionizing ECG signals analysis and Atrial Fibrillation/ Arrhythmias detection through the power of data science. In this endeavor, we focus on understanding and classifying heart ECG signals to detect underlying arrhythmias using random forest classifier.

<strong>Why AFib Detection Matters:</strong>

Arrhythmias are the major Cardiovascular diseases reported to be leading causes of deaths worldwide and Afib is the most prominent in that regards as reported by American Heart Association. It is therefore necessary to ddetect it early so as to know what direction of treatment to follow.</p>
<h4><b>
Our Approach:
</b>
</h4>

<p>
<b>Data</b> Collection: We gather extensive data on ecg signals from kaggle, physionet, and shelab to make a diverse database for model training..

<b>Data Preprocessing</b>: We cleaned and prepared the data for analysis, ensuring accuracy and reliability. Some of which included filtering, baseline correction, resampling, feature extraction extraction, etc.

<b>Random Forest</b>: Employing the Random Forest Classifier algorithm, we were able to classify ECG into 3 classes of  NSR, AFib and other arrhythmias with good metrics of 0.93 F1-score across all datasources.

Key Benefits:
<ul>
<li>Swift Diagnosis</li>
<li>Heart Parameters monitoring</li>
<li>Budget Efficiency</li>
<li>Improved health</li>
</p>
<h4>
Why Choose Our Solution:
</h4>
Our team of expert data scientists is dedicated to creating an ECG analysis tool for health monitoring. We are also working on hardwre integration to aid timely hart monitoing even more.
</p>
Get Started:

<i>
Join us on this journey of transforming cardiovascular health by harnessing the power of data. Unlock the potential of cardiovascular health monitoring with our cutting-edge solutions.
</i>
</p>
"""

def displaySignal(signal, column):
    fig = plt.figure(figsize = (10,10))
    plt.plot(signal[:2500])
    plt.title('ORIGINAL ECG DATA')
    column.pyplot(fig)

    fig2 = plt.figure(figsize = (15,15))
    out = ecg.ecg(signal=signal[:5000].flatten(), sampling_rate = 500, show=False)
    plotEcg(ts=out['ts'],
            raw=signal[:5000],
            filtered=out['filtered'],
            rpeaks=out['rpeaks'],
            heart_rate_ts=out['heart_rate_ts'],
            heart_rate=out['heart_rate'],
            column=column)

col1, col2, col3 = st.columns([1,4,1])

nav = st.sidebar.radio("Navigation",["Home", "Data Exploration","Prediction"])

if nav == "Home":

    st.markdown(html_temp, unsafe_allow_html=True)


if nav == 'Data Exploration':
    dataToExplore  = st.radio("Data to Explore",["AFIB DATA","NSR Data", "OTHERS"])
    signal = signals[dataToExplore]
    displaySignal(signal, col2)
    
if nav == "Prediction":
    st.header("PREDICT ECG CLASS")
    predType = st.radio("Select Prediction Type",["Upload Data","Predict Single"], horizontal = True)
    if predType == 'Upload Data':
        dat  = st.file_uploader("Upload your ECG data data. Supported formats are .csv, .txt, .mat, .ecg", type=['csv','txt','mat', '.npy'])
        if dat is not None:
            col2.write('PARSING DATA')
            if dat.name.endswith('.csv') or dat.name.endswith('.txt'): loadedData = pd.read_csv(dat).values.flatten()
            elif dat.name.endswith('.mat'): loadedData = loadmat(dat).flatten()
            elif dat.name.endswith('.npy'): loadedData = np.load(dat, allow_pickle = True).flatten()

            col2.write("PROCESSING DATA")
            if len(loadedData)>1000000: loadedData=loadedData[:1000000]
            arPeak    = ecg.ecg(loadedData, 500, show = False)['rpeaks']
            intervals = np.diff(arPeak)[1:]
            for i in range(len(intervals) - window_size + 1):
                intV       = intervals[i:i + window_size]
                extractedFeatures.append(extractFeatures(intV, fs = 500))
            extractedFeatures = np.array(extractedFeatures)
            displaySignal(loadedData, col2)
            
            if st.button("Predict"):
                    preds, probability = predict(data=extractedFeatures)
                    col2.write(f"THE GIVEN HEART RHYTHM IS OF {classes[preds]} WITH A CONFIDENCE OF {np.round(probability, 4)*100}%")


    if predType == 'Predict Single':
        val1 = col2.number_input( 'RMSSD',   0.0)
        val2 = col2.number_input( 'STDNN',   0.0)
        val3 = col2.number_input( 'MEAN_RR', 0.0)
        val4 = col2.number_input( 'MEAN_HR', 0.0)
        val5 = col2.number_input( 'STD_HR',  0.0)
        val6 = col2.number_input( 'MIN_HR',  0.0)
        val7 = col2.number_input( 'MAX_HR',  0.0)


        array = np.array([val1, val2, val3, val4, val5, val6, val7]).reshape(-1,7)
        if st.button("Predict"):
            preds, probability = predict(data=array)
            col2.write(f"THE GIVEN HEART FEATURES IS OF {classes[preds]} WITH A CONFIDENCE OF {np.round(probability, 4)*100}%")    
