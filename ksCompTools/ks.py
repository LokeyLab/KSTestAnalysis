# import ray
# ray.init(runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}})
# matplotlib==3.7.1 matplotlib-inline==0.1.6 numpy==1.24.1 pandas==2.0.2 pandocfilters==1.5.0 scikit-learn==1.2.2 scipy==1.10.1 seaborn==0.12.2 statsmodels==0.14.0 streamlit==1.25.0 virtualenv==20.24.1 plotly==5.15.0 kaleido==0.2.1 modin[dask]


import numpy as np
import modin.pandas as pd

from scipy.stats import kstest as kstest
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import re
import io
import sys
from tqdm.auto import tqdm


def dup(seed, duplicate=True, ID='median'):
    print('Removing Duplicates', file=sys.stderr)
    
    print(seed.shape, file=sys.stderr)
    if ID == 'median':
        df = seed.groupby(seed.index.name).median().reset_index()
    elif ID =='mean':
        df = seed.groupby(seed.index.name).mean().reset_index()
    elif ID =='min':
        df = seed.groupby(seed.index.name).min().reset_index()
    else:
        df = seed.groupby(seed.index.name).max().reset_index()
    
    seedIndex = df.iloc[:,0]

    df.set_index(seedIndex, inplace=True)
    df = df.iloc[:,1:]

    print("RETURING FROM dup TO CALL WITH DF:",df.shape, file=sys.stderr)
    return df

def generateEucDist(df: pd.DataFrame):
    return pd.DataFrame(squareform(pdist(df)), index=df.index, columns=df.index)

def ecdf(data):
    x = np.sort(data)
    y = np.arange(1,len(data)+1)/len(data)
    return x,y

def generateCorr(x: pd.DataFrame):
    return x.transpose().corr(method='pearson')

def calcKS(group, data, euc = True, forceCalc = False):
    '''YIELDS : name, p, xLims, ecdfData_i, ecdfFbk'''
    groupIn = group
    test = []

    for name, groups in tqdm(sorted(groupIn.items(), key=lambda a: a[0])):
        currGroup = [currNames for currNames in groups if currNames in data.index]
        data_i = data[[*currGroup]]
        #print(name, groups, currGroup)
        #print(data_i)
        data_i = data_i[data_i.index.isin(currGroup)]
        data_i = data_i.values
        data_i = np.triu(data_i, k=0)
        data_i = pd.DataFrame(data_i, index=currGroup, columns=currGroup)
        data_i.drop_duplicates(inplace=True)
        data_iFlat = np.array(data_i).flatten()
        data_iFlat = data_iFlat[data_iFlat != 0]
        #data_iFlat = np.unique(data_iFlat)
        
        fBk = data.drop(currGroup, axis=1, inplace=False)
        fBk = fBk[fBk.index.isin(currGroup)]
        fBk = np.array(fBk).flatten()
        #print(fBk)

        fBkdata_i = np.append(data_iFlat,fBk)
        #if forceCalc:
        #fBkdata_i = fBkdata_i[~np.isnan(fBkdata_i)]
        #fBkdata_i.fillna(0,inplace=True)
        
        #lims = [np.quantile(fBkdata_i, q=0.02), np.quantile(fBkdata_i, q=0.98)]

        try: 
            lims = [np.quantile(fBkdata_i, q=0.02), np.quantile(fBkdata_i, q=0.98)]
        except:
            print(fBkdata_i, file=sys.stderr)
            lims = [0,1]
        
        if len(data_iFlat) == 0 or len(fBk) == 0:
            if forceCalc:
                continue
            else:
                print(f'name = {name};;data_iFlat = {len(data_iFlat)};;fBk = {len(fBk)}')
                sys.exit(-1)

        if euc:
            p = kstest(data_iFlat, fBk, alternative='greater')[1]
            lims = [0,max(fBkdata_i)]
        else:
            p = kstest(data_iFlat, fBk, alternative='less')[1]
            lims = [-1,1]

        ecdfData_i = ecdf(data_iFlat)
        ecdfFBk = ecdf(fBk)

        yield name, p, lims, ecdfData_i, ecdfFBk

def generateECDF(pearsonData: list, eucData: list, groups: dict, names: list, forceCalc=False):
    '''
    EACH ECDF OBJECT CONTAINS: (dsName, [calcKS member])
    calcKS member:
        [(name, p, xLims, ecdfData_i, ecdfFBk),...,(...)]
    '''
    pearsonObjs = [(y, [i for i in calcKS(group=groups, data=x, euc=False, forceCalc=forceCalc)])\
                    for x, y in sorted(zip(pearsonData, names), key=lambda a:a[1])]
    eucObjs = [(y, [i for i in calcKS(group=groups, data=x, euc=True, forceCalc=forceCalc)])\
                    for x, y in sorted(zip(eucData, names), key=lambda a: a[1])]
    
    # if there are any problems, then check the arrays with nan
    print("RETURNING TO CALL FROM generateECDF: lengths of return are:", len(pearsonObjs),len(eucObjs),file=sys.stderr)
    return pearsonObjs, eucObjs

def plotData(pearsonData: list, eucData: list, groups: dict, names: list, pdfName, produceImg = True, forcePlot = False):
    assert len(pearsonData) == len(eucData) and len(names) == len(eucData)
    print("INSIDE ks.py/plotData:", len(pearsonData),len(eucData),file=sys.stderr,end="\n\n")
    pearsonObjs, eucObjs = generateECDF(pearsonData=pearsonData, eucData=eucData, groups=groups, names=names)
                    
    # Each calcKS obj contains: name, p, lims, ecdf for data_i, ecdf for fBk
    if produceImg:
        try:
            print(pearsonObjs,file=sys.stderr)
            assert len(pearsonObjs) == len(eucObjs) and len(pearsonObjs[0][1]) == len(eucObjs[-1][1])
        except:
            print(len(pearsonObjs), len(eucObjs), len(pearsonObjs[0][1]), len(eucObjs[-1][1]))
            print('some length is different')
            sys.exit(-1)
        # if my set-up is correct then all calcKs objs should have the same size
        

        lenDs = max(len(pearsonObjs[0][1]), len(eucObjs[0][1]))
        dataObjLen = len(eucData)

        p = PdfPages(pdfName)
        for i in range(lenDs):
            fig, ax = plt.subplots(2, dataObjLen, figsize=(4*dataObjLen, 10))
            for k, j in zip(range(len(ax[0])), range(dataObjLen)):
                try:
                    pearsonSubData = pearsonObjs[j][1][i]
                    euclideanSubData = eucObjs[j][1][i]
                except:
                    pass

                ax[0][k].step(pearsonSubData[3][0], pearsonSubData[3][1], \
                    marker='.', linestyle = None, markersize=2, color='red', rasterized=True)
                ax[0][k].step(pearsonSubData[4][0], pearsonSubData[4][1], \
                    marker='.', linestyle = None, markersize=2, color='blue', rasterized=True)

                ax[1][k].step(euclideanSubData[3][0], euclideanSubData[3][1], \
                    marker='.', linestyle = None, markersize=2, color='red', rasterized=True)
                ax[1][k].step(euclideanSubData[4][0], euclideanSubData[4][1], \
                    marker='.', linestyle = None, markersize=2, color='blue', rasterized=True)

                ax[0][k].set_xlim(pearsonSubData[2])
                ax[1][k].set_xlim(euclideanSubData[2])

                ax[0][k].set_title(f'{pearsonObjs[j][0]}\n p={pearsonSubData[1]:0.02e}')

                ax[1][k].set_title(f'p={euclideanSubData[1]:0.02e}')

                ax[0][0].set_ylabel('pearson R', fontsize=18)
                ax[1][0].set_ylabel('euclidean distance', fontsize=18)
            fig.suptitle(f'{pearsonObjs[0][1][i][0]}', fontsize=20)
            fig.tight_layout()
            fig.savefig(p, format='pdf')
        p.close()
        plt.close()

    return pearsonObjs, eucObjs
    
def getPvalues(eucData, pearsonData, groups):
    eucPVals = {}
    pearsonPVals = {}
    groupLen = len(groups)
    compOrder = []
    for i in range(len(eucData[0][1])):

        if len(compOrder) < groupLen:
            compOrder.append(eucData[0][1][i][0])

        for j in range(len(eucData)):
            currEuc = eucData[j][1][i]
            currPearson = pearsonData[j][1][i]
            groupKey = pearsonData[j][0]


            if groupKey not in eucPVals:
                eucPVals[groupKey] = np.array([currEuc[1]])
            else:
                eucPVals[groupKey] = np.append(eucPVals[groupKey], [currEuc[1]])
            
            if groupKey not in pearsonPVals:
                pearsonPVals[groupKey] = np.array([currPearson[1]])
            else:
                pearsonPVals[groupKey] = np.append(pearsonPVals[groupKey], [currPearson[1]])
    #print(eucPVals)
    for k,v in eucPVals.items():
        eucPVals[k] = -1 * np.log10(v)
    for k,v in pearsonPVals.items():
        pearsonPVals[k] = -1 * np.log10(v)

    return pd.DataFrame(eucPVals, index=compOrder), pd.DataFrame(pearsonPVals, index=compOrder)

def plotDataAppMod(pearsonData: list, eucData: list, selectComps: list = None):
    '''
    Modified version of original app data to make it more efficient for web app
    '''

    assert len(pearsonData) == len(eucData)
    pearsonObjs, eucObjs = pearsonData, eucData
    assert len(pearsonObjs) == len(eucObjs) and len(pearsonObjs[0][1]) == len(eucObjs[-1][1])
    # Each calcKS obj contains: name, p, lims, ecdf for data_i, ecdf for fBk

    # if my set-up is correct then all calcKs objs should have the same size
    lenDs = max(len(pearsonObjs[0][1]), len(eucObjs[0][1]))
    dataObjLen = len(eucData)

    for i in range(lenDs):
        if pearsonData[0][1][i][0] in selectComps:
            fig, ax = plt.subplots(2, dataObjLen, figsize=(5*dataObjLen, 10))
            for k, j in zip(range(len(ax[0])), range(dataObjLen)):
                try:
                    pearsonSubData = pearsonObjs[j][1][i]
                    euclideanSubData = eucObjs[j][1][i]
                except:
                    pass

                ax[0][k].step(pearsonSubData[3][0], pearsonSubData[3][1], \
                    marker='.', linestyle = None, markersize=2, color='red', rasterized=True)
                ax[0][k].step(pearsonSubData[4][0], pearsonSubData[4][1], \
                    marker='.', linestyle = None, markersize=2, color='blue', rasterized=True)

                ax[1][k].step(euclideanSubData[3][0], euclideanSubData[3][1], \
                    marker='.', linestyle = None, markersize=2, color='red', rasterized=True)
                ax[1][k].step(euclideanSubData[4][0], euclideanSubData[4][1], \
                    marker='.', linestyle = None, markersize=2, color='blue', rasterized=True)

                ax[0][k].set_xlim(pearsonSubData[2])
                ax[1][k].set_xlim(euclideanSubData[2])

                ax[0][k].set_title(f'{pearsonObjs[j][0]}\n p={pearsonSubData[1]:0.02e}')

                ax[1][k].set_title(f'p={euclideanSubData[1]:0.02e}')

                ax[0][0].set_ylabel('pearson R', fontsize=18)
                ax[1][0].set_ylabel('euclidean distance', fontsize=18)
            fig.suptitle(f'{pearsonObjs[0][1][i][0]}', fontsize=20)
            fig.tight_layout()
            yield fig

###########################################
# TWEAK Figure Sizes Here
###########################################
def jitter(arr):
    stdev = 0.005 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def plotScatter(df: pd.DataFrame, ax: plt.Axes, xName: str, size=100):
    minY = min(df.min(axis=1))
    maxY = max(df.max(axis=1))
    # minY = 0
    # maxY = 5

    xNames = list(df.index)
    x = [i for i in np.arange(len(xNames))]
    for col in df.columns:
        ax.scatter(x=jitter(x), y=df[col], s=size, rasterized = True)
    
    fs = (len(df.index)*2)
    ax.grid(axis='x')
    ax.xaxis.set_ticks(list(np.arange(0,df.shape[0])))
    ax.xaxis.set_ticklabels(list(df.index), fontsize = 10)
    ax.set_ylim(minY, maxY)
    
    ax.tick_params(axis='x', labelrotation=90, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    ax.axhline(y=-1*np.log10(0.01), linewidth=2, linestyle='--', color='gray')
    ax.axhline(y=-1*np.log10(0.05), linewidth=2, linestyle='--', color='gray')

    ys = (len(df.index)*5)
    ax.legend(list(df.columns), loc='upper left', fontsize=10)
    ax.set_xlabel(xName, fontsize=50)
    ax.set_ylabel('-log10 P values', fontsize=10)

def plotPVals(pearsonDf: pd.DataFrame, euclideanDf: pd.DataFrame):
    assert pearsonDf.shape == euclideanDf.shape
 
    x = list(pearsonDf.index)
    xn = [i for i in np.arange(len(x))]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pearson R", 'Euclidean Distance'), shared_yaxes=True, y_title='-log10 P Values')

    for col in pearsonDf.columns:
        fig.add_trace(
            go.Scatter(x = x, y = pearsonDf[col], mode = 'markers', name = f'{col}_PearsonR'),
            row=1,
            col=1
        )
        fig.add_trace(
            go.Scatter(x = x, y = euclideanDf[col], mode = 'markers', name = f'{col}_EuclideanDist'),
            row=1,
            col=2
        )

    fig.add_hline(y=-1*np.log10(0.05), row=1, line_dash = 'dash')
    fig.add_hline(y=-1*np.log10(0.01), row=1, line_dash = 'dash')

    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = xn,
            ticktext = x,
            tickangle = 90,
            title = 'Compound Classes'
        ),
        xaxis2 = dict(
            tickmode = 'array',
            tickvals = xn,
            ticktext = x,
            tickangle = 90,
            title = 'Compound Classes'
        ),
        title_text = '-log10 P Values'
    )
    return fig

def __plotLogBar(df: pd.DataFrame, log: dict, fig: go.Figure, row: int, col: int):
    y = [v for k,v in sorted(log.items(), key=lambda a: a[0])]
    x = [k for k,v in sorted(log.items(), key=lambda a: a[0])]
    yPercent = [f'{i/len(df)*100:0.1f}%' for i in y]

    fig.add_trace(
        go.Bar(x=x, y=y),
        row=row, col=col
    )
    fig.update_traces(texttemplate = yPercent, textposition = 'inside', row=row, col=col, showlegend=False)

def plotClassesAboveThreshold(df: pd.DataFrame, name: str):
    log5Classes = {}
    log1classes = {}

    for col in df.columns:
        countLog1 = len(df[df[col] > (-1 * np.log10(0.01))].index)
        countLog5 = len(df[df[col] > (-1 * np.log10(0.05))].index)
        log5Classes[col] = countLog5
        log1classes[col] = countLog1

    fig = make_subplots(
        rows=1,cols=2,
        subplot_titles=('Classes above -log10(0.01) threshold per dataset', 'Classes above -log10(0.05) threshold per dataset')
    )
    __plotLogBar(df=df, log=log1classes, fig=fig, row=1, col=1)
    __plotLogBar(df=df, log=log5Classes, fig=fig, row=1, col=2)

    fig.update_layout(
        yaxis = dict(
            title = 'Class counts'
        ),
        yaxis2 = dict(
            title = 'Class counts'
        ),
        xaxis = dict(
            title = 'Datasets',
            tickangle = 90
        ),
        xaxis2 = dict(
            title = 'Datasets',
            tickangle = 90
        ),
        title = dict(
            text = f'Number of Classes above -log10 threshold for {name}',
            x = 0.5
        )
    )

    return fig
