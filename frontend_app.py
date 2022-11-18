import streamlit as st
import pandas as pd
import requests

st.title('Programming Task for DKNSB Medical Device Consultants')

st.write('This simple frontend was built to create an interactive testing environment for the backend of the programming task.')

#file upload
st.write('Upload a pdf file to test the backend.')
uploaded_file = st.file_uploader("Choose a file", type="pdf")

#when file is uploaded, make post request to backend
if uploaded_file is not None:
    files = {'pdf': uploaded_file.getvalue()}
    response = requests.post('http://localhost:8000/pdf', files=files)
    print(response.json())

    #display results
    st.write('Results:')
    df = pd.DataFrame(response.json()['message'])
    st.write(df)

#upload image
st.write('Upload an image to test the backend.')
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png"])

#when image is uploaded, make post request to backend
if uploaded_image is not None:
    files = {'image': uploaded_image.getvalue()}
    response = requests.post('http://localhost:8000/image', files=files)
    print(response.json())

    #display results
    st.write('Results:')
    df = pd.DataFrame(response.json()['message'], index=[0])
    st.write(df)