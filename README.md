# Kolmogorov–Smirnov Test Analysis Web App
### R code originally developed by Akshar Lohith 
### Converted and modified into Python by Derfel Terciano

This is a web app that generates all analytics for the Kolmogorov–Smirnov Test when
given multiple datasets that share a similar compounds by a given key.

The Web App was developed in python while the front-end was developed with
Streamlit, a python front-end development library. 

The web app contains analytics dispayed as bar plots, scatter plots, and step plots.
Additionally, the main most of the displayed images are downloadable from the web app.

## If you would like a demo, you can try the web app on the [Streamlit website here](https://kstestanalysis.streamlit.app/).
*NOTE:* Due to the limited resources that the Streamlit cloud provides, **we do not recommend that you use the tool to its fullest extent on the**
**Streamlit website**. Instead, we recommend running the web app locally through a docker image (*instructions provided below*). Trying to use
the app to its fullest extent on the streamlit site will cause the program to crash. 
    - *If the app is not up and running on the link above please email <dtvsworld@gmail.com>.*