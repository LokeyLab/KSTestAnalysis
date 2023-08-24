import os, subprocess, io
from ksCompTools import *

import pandas as pd, numpy as np, plotly as plt

class KSProcessingCLI:
    def __init__(self, key: pd.DataFrame, dataFolder: str, keyTargs: list, exclude: list = None, dsNames: list = None):

        #initializing data
        self.key = key
        self.dataFolder = dataFolder
        self.exclude = exclude
        self.keyTargs = keyTargs
        self.data = {}

        # parse data Folder
        if dsNames is None:
            for file in os.listdir(self.dataFolder):
                currDataset = os.path.join(self.dataFolder, file)
                self.data[file] = pd.read_csv(currDataset, index_col=0)
                self.data[file] = dup(self.data[file])
        else:
            for name, file in zip(dsNames, os.listdir(self.dataFolder)):
                currDataset = os.path.join(self.dataFolder, file)
                self.data[name] = pd.read_csv(currDataset, index_col=0)
                self.data[name] = dup(self.data[name])

        self.datasetCorr = {k:generateCorr(v) for k,v in self.data.items()}
        self.datasetDist = {k:generateEucDist(v) for k,v in self.data.items()}
        assert len(self.datasetCorr) == len(self.datasetDist)

        # determine groups
        self.groups = key[self.keyTargs[0]].value_counts(dropna=True, ascending=False)
        self.groups = self.groups[self.groups > 3]
        if self.exclude is not None:
            self.groups = self.groups[~self.groups.index.isin(self.exclude)]
        self.groups = list(self.groups.index)

        #get final compounds and group ids
        self.comps = key[[*self.keyTargs]]
        self.comps = self.comps[self.comps.iloc[:,0].isin(self.groups)]
        self.comps = self.comps.groupby(self.comps.iloc[:,0])[self.comps.columns[1]]
        self.comps = {k: np.array(v) for k,v in self.comps}

        # generate useable dataets
        self.names = [k for k in sorted(self.datasetCorr.keys())]
        self.datasetsCorrList = [v for k,v in sorted(self.datasetCorr.items(), key=lambda a:a[0])]
        self.datasetsDistList = [v for k,v in sorted(self.datasetDist.items(), key=lambda a:a[0])]

        # plotting Data
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

    def plotCalcData(self, outName: str, selectComps: list = None):
        selects = self.groups if selectComps is None else selectComps
        self.pearsonECDF, self.euclideanECDF = plotData(pearsonData=self.datasetsCorrList, eucData=self.datasetsDistList, groups=self.comps, names=self.names, pdfName=outName)
    
    def generatePVals(self):
        pearson, euc = getPvalues(eucData=self.euclideanECDF, pearsonData=self.pearsonECDF, groups=self.groups)
        return pearson, euc
    
    def plotPVals(self, pearson, euc, outName, **kwargs):
        fig = plotPVals(pearsonDf=pearson, euclideanDf=euc)
        fig.write_image(f'{outName}PValPlot.pdf', format='pdf', **kwargs)


class CommandLine:
    def __init__(self, inOpts = None):
        from argparse import ArgumentParser

        self.parser = ArgumentParser(
            description='CLI version of KSTestApp program',
            prefix_chars='-',
            add_help=True,
            usage='python3 ksCLI.py -k <.csv file> -d <FOLDER containing datasets> -kt <list of string of key columns> -e <list of rows to exclue>'
        )

        #arguments
        self.parser.add_argument('-k', '--keyFile', type=str, required=True, nargs='?', action='store', help='Key file (.csv)')
        self.parser.add_argument('-d', '--datasets', type=str, required=True, nargs='?', action='store', help='A path to a folder containing all the datasets (.csv)')
        self.parser.add_argument('-kt', '--keyTarg', required=True, nargs='+', action='store', help='a list of strings that contains the 1) compound 2) IDs')
        self.parser.add_argument('-e', '--exclude', required=False, default=None, nargs='+', action='store', help='a list of strings that define which rows to exclude')
        self.parser.add_argument('-o', '--outName', required=False, type=str, default='out', nargs='?', action='store', help='name for outputs')
        self.parser.add_argument('-n', '--name', required=False, type=str, nargs='+', action='store', default=None, help='Rename datasets in order of how they appear in your directory')

        #args
        if inOpts is None:
            self.args = self.parser.parse_args()
        else:
            self.args = self.parser.parse_args(inOpts)

def main(inOpts = None):
    cl = CommandLine(inOpts=None)

    keyFile = cl.args.keyFile
    dsFolder = cl.args.datasets
    keyTargs = cl.args.keyTarg
    excludes = cl.args.exclude
    outFile = cl.args.outName

    key = pd.read_csv(keyFile, sep=',')

    ks = KSProcessingCLI(key=key, dataFolder=dsFolder, keyTargs=keyTargs, exclude=excludes, dsNames=cl.args.name)
    
    pdf = f'{outFile}ecdfGraphs.pdf'
    ks.plotCalcData(outName=pdf)
    pearson, euc = ks.generatePVals()

    pearson.to_csv(f'{outFile}Pearson.csv', sep =',')
    euc.to_csv(f'{outFile}Euc.csv', sep =',')

    ks.plotPVals(pearson=pearson, euc=euc, outName=outFile, width=1080, height=720)

if __name__ == '__main__':
    main()