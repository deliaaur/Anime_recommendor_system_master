import streamlit as st
import joblib
import pandas as pd
from PIL import Image

@st.cache(allow_output_mutation=True)
def load(model_path):
    df = pd.read_pickle(model_path)
    return df

def inference(df, name):
    similar_to_movie = df.corrwith(df[name])
    similar_to_movie = pd.DataFrame(similar_to_movie,columns=['Correlation'])
    similar_to_movie = similar_to_movie.sort_values(by = 'Correlation', ascending = False)
    return similar_to_movie

st.title('Anime Recommendation App')
st.write('Based on your favs we will give you another 20 anime to binge watch')
image = Image.open('data/TitleImage.png')
st.image(image, use_column_width=True)
dataframe = load('models/df.zip')

option = st.selectbox('Please select your favorite anime', (dataframe.columns))

st.write('You selected:', option)

if (st.button('Get Recommendation')):
    # dataframe = load('../models/df.pkl')
    result = inference(dataframe,option)
    st.write(result.head(20))
    st.balloons()

