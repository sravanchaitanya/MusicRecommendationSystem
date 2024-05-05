import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib
import requests
from PIL import Image
from io import BytesIO


def find_song(name, year):
    song_data = defaultdict()
    results = st.session_state.sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = st.session_state.sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    
    except IndexError:
        return find_song(song['name'], song['year'])
        

def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[st.session_state.number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict


def recommend_songs( song_list, spotify_data, n_songs=10):
    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = st.session_state.song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[st.session_state.number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

def get_image(name,year):
    result = st.session_state.sp.search(q = 'name: {} year: {}'.format(name,year))
    if not result['tracks']['items']==[]:
        url= result['tracks']['items'][0]['album']['images'][0]['url']
        return Image.open(BytesIO(requests.get(url).content))
    else:
        url= 'default.png'
        return Image.open(url)

    
def display_recommendation(songs,data):
    pointer = 0;
    columns = st.columns(4)
    for dic in songs:
        with columns[pointer%4]:
            st.image(get_image(dic['name'],dic['year']))
            st.write(dic['name'])
        pointer+=1


if 'initialized' not in st.session_state:
    top100 = pd.read_csv('C:/Users/srava/Desktop/Final_Project/data/top100.csv')
    st.session_state.top100 = top100
    st.session_state['curr_selection_list'] = list(top100['name'])
    print('yes')

form = st.form(key = 'start',clear_on_submit = True)        
selection = form.multiselect("Select the songs",st.session_state.curr_selection_list) 
Recommend = form.form_submit_button('Recommend')

if 'displayed' in st.session_state and not st.session_state.displayed:
    display_recommendation(st.session_state.recommended_songs,st.session_state.data)
    st.session_state.displayed=True

if Recommend:
    print('yaa')
    lis = []
    for dic in st.session_state.curr_song_list:
        if dic['name'] in selection:
            dic1 = {}
            dic1['name'] = dic['name']
            dic1['year'] = dic['year']
            lis.append(dic1)
    recommended_songs = recommend_songs(lis,st.session_state.data)
    st.session_state.curr_song_list=recommended_songs
    lis = []
    for dic in recommended_songs:
        lis.append(dic['name'])
    st.session_state.curr_selection_list=lis
    st.session_state.recommended_songs = recommended_songs
    st.session_state['displayed'] = False
    st.experimental_rerun()

if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    
    data = pd.read_csv("C:/Users/srava/Desktop/Final_Project/data/data.csv")
    st.session_state.data = data
    
    
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="6239ea24c3d64de2a09122ee3d157247",
                                                               client_secret="d4c17019488349318a3776b72f60b2aa"))
    st.session_state.sp = sp
    
    number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
     'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
    st.session_state.number_cols = number_cols

    song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                      ('kmeans', KMeans(n_clusters=20, 
                                       verbose=False))
                                     ], verbose=False)
    st.session_state.song_cluster_pipeline = song_cluster_pipeline
    
    X = data.select_dtypes(np.number)
    song_cluster_pipeline.fit(X)
    song_cluster_labels = song_cluster_pipeline.predict(X)
    data['cluster_label'] = song_cluster_labels
    
    lis = []
    for index in top100.index:
        dic = {}
        dic['name'] = top100['name'][index]
        dic['year'] = top100['year'][index]
        lis.append(dic)
    st.session_state['curr_song_list'] = lis
    display_recommendation(lis,st.session_state.data)
