import sys
import os
import io
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ksCompTools import *

st.set_page_config(layout='wide')

header = st.container()
body = st.container()
generator = st.container()

keyDF = None
dataSets = {}

def resertClicks():
    st.session_state.genClick = False
    st.session_state.click_load = False

with header:
    st.markdown("""
    ## Kolmogorov–Smirnov Test Analysis

    ##### To begin, please upload the required files in the ***side bar***
    *WARNING: This tool does not clean up input files so be sure that the IDs
    in the key file matches with the IDs on the datset files (i.e this tool 
    is very case sensitive)*
    """)


with st.sidebar:
    st.title('File Upload')
    key = st.file_uploader(label='Key File', type='.csv', accept_multiple_files=False)
    dataIn = st.file_uploader(label='Data set files', type='.csv', accept_multiple_files=True)

    if key is not None and dataIn is not None:
        keyDF = pd.read_csv(key, sep=',')
        for i in range(len(dataIn)):
            name = ''.join(dataIn[i].name.split('.')[:-1])
            dataSets[name] = pd.read_csv(dataIn[i], sep=',')
            dataSets[name].reset_index(drop=True, inplace=True)
            dataSets[name].set_index(dataSets[name].iloc[:,0], inplace=True)
            dataSets[name] = dataSets[name].iloc[:,1:]
    # st.button(label='Reset', on_click=reset())
    # st.cache_resource.clear()
    rs = st.button('Reset')
    if rs:
        resertClicks()

        

############
### MAIN ###
############
def resertClicks():
    st.session_state.genClick = False
    st.session_state.click_load = False
    st.session_state.pValGen = False
    st.session_state.pGen = False

def click_button():
    st.session_state.click_load = True

def click_genButton():
    st.session_state.genClick = True

@st.cache_resource
def initKS(keyIn, ds, keyT, exclude):
    ks = KSProcessing(key=keyIn, data=ds, keyTargs=keyT, exclude=exclude)
    calcs = ks.generateObjs()
    return ks, calcs

@st.cache_data
def dl_csv(df):
    return df.to_csv().encode('utf-8')

ks, calcs = None, None

if keyDF is not None and len(dataSets) > 0:
    with body:
        st.markdown(
            """
            ### Output options:
            """
        )
        
        leftCol, rightCol = st.columns(2)
        with leftCol:
            compoundKey = st.selectbox('Select the compound Class column from key file:', keyDF.columns)
        
        with rightCol:
            idKey = st.selectbox('Select the compound ID columns from the key file:', keyDF.columns)

        userCompoundOptions = keyDF[compoundKey].unique()
        excluded = st.multiselect(label='Select compounds to exclude if any:', options=userCompoundOptions, \
            placeholder='Choose compound(s)')

        keyTargs = [compoundKey, idKey]

        nameChangeCheckBox = st.expander(label='Edit Dataset Names')
        with nameChangeCheckBox:
            names = [k for k in sorted(dataSets.keys())]
            #changeName = st.container()
            changeName = st.form("Change Name of Datasets")
            with changeName:
                for i, k in enumerate(sorted(names)):
                    newIn = st.text_input(label=k, value=k)
                    if newIn != k and newIn != '':
                        dataSets[newIn] = dataSets[k]
                        del dataSets[k]
                    #st.write('changed to:',test)
                submitButt = st.form_submit_button('Change Dataset Display names')

    with generator:
        if 'data' not in st.session_state:
            st.session_state.data = []
        st.markdown(
            '''
            ### KS Comparison
            '''
        )

        if 'click_load' not in st.session_state:
            st.session_state.click_load = False
        if 'genClick' not in st.session_state:
            st.session_state.genClick = False
        
        loadOk = st.button(label='Load Data')
        # clickedForced = False
        # genPlotForce = False
        if loadOk: 
            st.session_state.genClick = False
            click_button()

        if st.session_state.click_load == True:

            ex = None if len(excluded) == 0 else excluded
            ks, calcs = initKS(keyIn=keyDF, ds=dataSets, keyT=keyTargs, exclude=excluded)
            st.session_state.data = [ks, calcs]
            ks, calcs = st.session_state.data

            st.success('Data has been successfully loaded!', icon='✅')
            currGroups = ks.getGroups
            tabName = ['Generate Plots', 'Generate Log10 P-Values']
            plotTab, pValTab = st.tabs(tabName)

            with plotTab:
                plotGenForm = st.form('plotGen')
                with plotGenForm:
                    selects = st.multiselect(label='Choose which compounds to display (empty means all compounds will be displayed)', options=currGroups)
                    c1, c2 = st.columns(2)
                    with c1:
                        submitted = st.form_submit_button("Generate Plot")
                    with c2:
                        dl = st.form_submit_button("Download Plots")
                
            
                if submitted and not dl:
                    click_genButton()
                    selects = None if len(selects) == 0 else selects
                elif dl and not submitted:
                    st.session_state.genClick = False
                    
                    progText = 'Generating pdf file'

                    with st.spinner(progText):
                        selects = None if len(selects) == 0 else selects
                        f = io.BytesIO()
                        p = PdfPages(f)
                        for fig in ks.plotProcessedData(selectComps=selects):
                            i += 1
                            fig.savefig(p, format = 'pdf')
                            fig.clf()
                            plt.figure().clear()
                            plt.close()
                            plt.cla()
                            plt.clf()
                        p.close()

                    progBar = st.success('Finished!')

                    dl = st.download_button(
                        label='Download plots',
                        data=f,
                        mime='image/pdf',
                        file_name='tool.pdf'
                    )

                if st.session_state.genClick:
                    with st.spinner('Displaying plots...'):
                    #figs = [x for x in ks.plotProcessedData()]
                    #print(figs)
                        dl = st.empty()
                        for fig in ks.plotProcessedData(selectComps=selects):
                            st.pyplot(fig)
                    st.success('Displayed plots!')
            
            with pValTab:
                if 'pGen' not in st.session_state:
                    st.session_state.pGen = False

                genPVal = st.button('Generate P-values')

                pearson, euc = None, None
                if genPVal:
                    with st.spinner('Generating Data'):
                        pearson, euc = ks.generatePVals()
                    st.success('Data successfully generated!')
                    st.session_state.pGen = True

                if pearson is not None and euc is not None and st.session_state.pGen:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write('Pearson -Log10 P-Values')
                        st.dataframe(pearson, use_container_width=True)
                        st.download_button(
                            label= ' Download Pearson -log10 R values as csv',
                            data = dl_csv(pearson),
                            file_name='pearson.csv',
                            mime='text/csv'
                        )

                    with c2:
                        st.write('Euclidean -Log10 P-Values')
                        st.dataframe(euc, use_container_width=True)
                        st.download_button(
                            label= ' Download Euclidean -log10 R values as csv',
                            data = dl_csv(euc),
                            file_name='euclidean.csv',
                            mime='text/csv'
                        )

                    st.markdown(
                        """
                        ##### -log10 P value plots
                        """
                    )

                    ##########Plots
                    pFig = ks.plotPValues(pearson=pearson, euc=euc)
                    st.plotly_chart(pFig, use_container_width=True, theme=None)

                    st.markdown(
                        """
                        ##### Plots for the number of classes that are above the -log10(0.001) and -log10(0.005) threshold
                        """
                    )

                    c1, c2 = st.columns(2)
                    with c1:
                        #img = io.BytesIO()
                        tFig = ks.plotThresholds(pearson, 'Pearson R')
                        st.plotly_chart(tFig, theme=None)
                        
                    with c2:
                        # img = io.BytesIO()
                        tFig = ks.plotThresholds(euc, 'Euclidean')
                        st.plotly_chart(tFig, theme=None)
                        

                    # st.markdown(
                    #     """
                    #     ##### -log10 P value tables
                    #     """
                    # )
                
                

else:
    resertClicks()
