'''
A processing procedure for the KS comparison web app
'''
import sys, os

from ksCompTools.ks import *
import numpy as np
import pandas as pd
import io

class KSProcessing:
    '''KS Processing for Webapp'''
    def __init__(self, key: pd.DataFrame, data: dict, keyTargs: list, exclude: list = None):
        
        #initializing data
        self.key = key
        self.data = {k:dup(v) for k,v in data.items()}
        self.keyTargs = keyTargs # note keyTargs[0] = compound; keyTargs[1] = IDs 
        self.exclude = exclude # list of compounds that should be excluded

        #generating appropriate correlations and distances
        self.datasetsCorr = {k:generateCorr(v) for k,v in self.data.items()}
        self.datasetsDist = {k:generateEucDist(v) for k,v in self.data.items()}
        assert len(self.datasetsCorr) == len(self.datasetsDist)

        # determining groups
        self.groups = key[self.keyTargs[0]].value_counts(dropna=True, ascending=False)
        self.groups = self.groups[self.groups > 3]
        if self.exclude is not None:
            self.groups = self.groups[~self.groups.index.isin(self.exclude)]
        self.groups = list(self.groups.index)

        # getting the final compounds and their grouped ids
        self.comps = key[[*self.keyTargs]]
        self.comps = self.comps[self.comps.iloc[:,0].isin(self.groups)]
        self.comps = self.comps.groupby(self.comps.iloc[:,0])[self.comps.columns[1]]
        self.comps = {k: np.array(v) for k,v in self.comps}

        # generate useable datasets
        self.names = [k for k in sorted(self.datasetsCorr.keys())]
        self.datasetsCorrList = [v for k,v in sorted(self.datasetsCorr.items(), key=lambda a:a[0])]
        self.datasetsDistList = [v for k,v in sorted(self.datasetsDist.items(), key=lambda a:a[0])]

        # used to store plotting data for plotting function
        self.pearsonECDF = None
        self.euclideanECDF = None
        self.selectedComps = None

    @property
    def getGroups(self):
        return self.groups
    
    def generateObjs(self, selectComps: list = None):
        if selectComps is not None:
            compsSelected = {k:v for k,v in self.comps.items() if k in selectComps}
        else:
            compsSelected = self.comps
        
        pearson, euc = generateECDF(self.datasetsCorrList, self.datasetsDistList, groups=compsSelected, names=self.names)

        self.selectedComps = compsSelected
        self.pearsonECDF, self.euclideanECDF = pearson, euc
        return pearson, euc
    
    def plotProcessedData(self, selectComps: list = None):
        selects = self.groups if selectComps is None else selectComps
        return plotDataAppMod(self.pearsonECDF, self.euclideanECDF, selectComps=selects)
    
    def generatePVals(self):
        pearson, euc = getPvalues(eucData=self.euclideanECDF, pearsonData=self.pearsonECDF, groups=self.groups)
        return pearson, euc
    
    def plotPValues(self, pearson, euc):
        return plotPVals(pearson, euc)
    
    def plotThresholds(self, df, name):
        return plotClassesAboveThreshold(df, name)

