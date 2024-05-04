import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import requests
import builtins
from PIL import Image
from io import BytesIO

top100 = pd.read_csv('C:/Users/srava/Desktop/Final_Project/data/top100.csv')

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="6239ea24c3d64de2a09122ee3d157247",
                                                           client_secret="d4c17019488349318a3776b72f60b2aa"))



def get_image(name,year):
    result = sp.search(q = 'name: {} year: {}'.format(name,year))
    if not result['tracks']['items']==[]:
        url= result['tracks']['items'][0]['album']['images'][0]['url']
        return Image.open(BytesIO(requests.get(url).content))
    else:
        url= 'default.png'
        return Image.open(url)

images=[]

for i in range(0,100):
    images.append(get_image(top100['name'][i],top100['year'][i]))
    
containers = []

columns = st.columns(3)

for i in range(0,100):
    with columns[i%3]:
        container = st.container()
        containers.append(container)
        with container:
            st.image(images[i])
            st.write(top100['name'][i])
            flag = st.button('Select',key=i)
            
