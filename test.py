import streamlit as st
from sklearn.datasets import fetch_california_housing 
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestRegressor
st.header('This is my first app')
st.text('Polayadi mone')
housing = fetch_california_housing()
df = pd.DataFrame(housing.data,columns = housing.feature_names)
df['target']=housing .target
st.write(df.head)
st.write(df.shape)
st.write(df.columns)
st.text("Information about the dataset - missing values")
st.write(df.isnull().sum())

chart = st.container()
model = st.container()
predict = st.container()

with chart:
    st.text("This is inside container")
    st.bar_chart(df['Population'])
    st.area_chart(df['Latitude'])
    st.line_chart(df['HouseAge'])
    st.scatter_chart(df['target'])

with model:
    X = df.drop("target", axis=1)  # Replace "target_column" with actual name
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()  # Replace with your model
    model.fit(X_train,y_train)