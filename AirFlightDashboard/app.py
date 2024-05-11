import streamlit as st
from dbHelper import DB
import plotly.graph_objects as go
import plotly.express as px

db  = DB()

st.sidebar.title("Flight Analytics")
user_options = st.sidebar.selectbox("Menu",["Select One","Check Flights","Analytics"])

if(user_options == "Check Flights"):
    st.title("Check Flights")
    col1,col2 = st.columns(2)
    with col1:
        city = db.fetch_city_names()
        source = st.selectbox("Source",sorted(city))
    with col2:
        city = db.fetch_city_names()
        destination = st.selectbox("Destination",sorted(city))
    
    if(st.button("Search")):
        results = db.fetch_all_flights(source,destination)
        st.dataframe(results)

elif(user_options == "Analytics"):
    st.title("Analytics")
    airline,frequency = db.fetch_airline_frequency()
    fig = go.Figure(go.Pie(labels=airline,values=frequency,hoverinfo="label+percent",textinfo="value"))
    st.header("Pie Chart")
    st.plotly_chart(fig)

    city,frequency1 = db.busy_airport()
    fig = px.bar(x = city,y = frequency1)
    st.header("Bar Chart")
    st.plotly_chart(fig,theme="streamlit",use_container_width = True)

    date,frequency2 = db.daily_frequency()
    fig = px.line(x = date,y = frequency2)
    st.header("Line Chart")
    st.plotly_chart(fig,theme="streamlit",use_container_width = True)
else:
    st.title("Tell about the project")