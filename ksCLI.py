import os, subprocess, io, glob, pathlib, sys
import tqdm.auto as tqdm
from ksCompTools import *

import pandas as pd, numpy as np, plotly as plt

def ensure_path_exists(file_path: pathlib.Path) -> pathlib.Path:
    # Convert the input to a Path object
    path = pathlib.Path(file_path)
   
    # Check if the directory of the path exists, and if not, create it
    path.mkdir(parents=True, exist_ok=True)

    return path

class KSProcessingCLI:
    def __init__(self, key: pd.DataFrame, dataFolder: str, keyTargs: list, exclude: list = None, dsNames: dict = None, pickle: bool = False,read_pickle: bool = False):

        #initializing data
        self.key = key
        self.dataFolder = dataFolder
        self.exclude = exclude
        self.keyTargs = keyTargs
        self.data = {}
        
        # determine groups
        self.groups = self.key[self.keyTargs[0]].value_counts(dropna=True, ascending=False)
        self.groups = self.groups[self.groups > 3]
        if self.exclude is not None:
            self.groups = self.groups[~self.groups.index.isin(self.exclude)]
        self.groups = list(self.groups.index)

        #get final compounds and group ids
        self.comps = self.key[[*self.keyTargs]]
        self.comps = self.comps[self.comps.iloc[:,0].isin(self.groups)]
        self.comps = self.comps.groupby(self.comps.iloc[:,0])[self.comps.columns[1]]
        self.comps = {k: np.array(v) for k,v in self.comps}
                
        if not read_pickle:
            print(f"reading datafile from {self.dataFolder}",file=sys.stderr)
            # parse data Folder
            if dsNames is None:
                for file in tqdm(glob.glob(self.dataFolder+"/*.csv")):
                    currDataset = os.path.join(self.dataFolder, os.path.basename(file))
                    dictname = os.path.basename(currDataset).replace('.csv','')
                    print(f"reading dataset {dictname}",file=sys.stderr)
                    self.data[dictname] = pd.read_csv(currDataset, index_col=0)
                    if self.data[dictname].index.name is None:
                        self.data[dictname].index.name = 'Index'
                    self.data[dictname] = dup(self.data[dictname])
                    df = self.data[dictname]
                    # clear datasets of rows not present in the key
                    self.data[dictname] = df.loc[df.index.isin(self.key[self.keyTargs[1]].to_list())]
                    del df
            else:
                for name, file in tqdm(dsNames.items()):
                    #, glob.glob(self.dataFolder+"/*.csv"))):
                    currDataset = os.path.join(self.dataFolder,file)
                    print(f"Reading dataset {file} as {name}",file=sys.stderr)
                    self.data[name] = pd.read_csv(currDataset, index_col=0)
                    if self.data[name].index.name is None:
                        self.data[name].index.name = 'Index'
                    self.data[name] = dup(self.data[name])
                    df = self.data[name]
                    # clear datasets of rows not present in the key
                    self.data[name] = df.loc[df.index.isin(self.key[self.keyTargs[1]].to_list())]
                    del df
                    
            self.datasetCorr = {k:generateCorr(v) for k,v in self.data.items()}
            self.datasetDist = {k:generateEucDist(v) for k,v in self.data.items()}
            
            # once the nxn correlation and distmats are computed
            # OG dfs not needed, keep the df_names (keys)
            self.data = self.data.keys()
            
        else:
            print(f"reading pickled simMats from {os.path.join(self.dataFolder,'pickles')}",file=sys.stderr)
            # read pickled simMats from pickle folder inside data folder
            # dsNames ignored, only use pickle names
            self.datasetCorr = dict()
            self.datasetDist = dict()
            for pickleFile in tqdm(glob.glob(self.dataFolder+"/pickles/*.hd5")):
                file = pickleFile.replace('.hd5','')
                if file.endswith('_CorrMat'):
                    self.datasetCorr[file.replace('_CorrMat','')] = pd.read_hdf(pickleFile,'corrDF')
                elif file.endswith('_DistMat'):
                    self.datasetDist[file.replace('_DistMat','')] = pd.read_hdf(pickleFile,'distDF')

        if pickle:
            picklePath = ensure_path_exists(os.path.join(self.dataFolder,"pickles"))
            [corrDF.to_hdf(os.path.join(picklePath , f"{k}_CorrMat.hd5"),key='corrDF',format='table',complevel=9,mode='w') for k,corrDF in tqdm(self.datasetCorr.items())]
            [distDF.to_hdf(os.path.join(picklePath , f"{k}_DistMat.hd5"),key='distDF',format='table', complevel=9,mode='w') for k,distDF in tqdm(self.datasetDist.items())]
            
        assert len(self.datasetCorr) == len(self.datasetDist)

        # generate useable datasets
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

    def plotCalcData(self, outName: str, selectComps: list = None, produceImg = True):
        selects = self.groups if selectComps is None else selectComps
        self.pearsonECDF, self.euclideanECDF = plotData(pearsonData=self.datasetsCorrList, eucData=self.datasetsDistList, groups=self.comps, names=self.names, pdfName=outName, produceImg=produceImg)
    
    def generatePVals(self):
        pearson, euc = getPvalues(eucData=self.euclideanECDF, pearsonData=self.pearsonECDF, groups=self.groups)
        return pearson, euc
    
    def plotPVals(self, pearson, euc, outName, **kwargs):
        fig = plotPVals(pearsonDf=pearson, euclideanDf=euc)
        fig.write_image(f'{outName}PValPlot.pdf', format='pdf', **kwargs)


class CommandLine:
    def __init__(self, inOpts = None):
        from argparse import ArgumentParser, FileType

        self.parser = ArgumentParser(
            description='CLI version of KSTestApp program',
            prefix_chars='-',
            add_help=True,
            usage='python3 ksCLI.py -k <.csv file> -d <FOLDER containing datasets> -kt <list of header strings of key columns> -e <list of classes to exclue>'
        )

        #arguments
        self.parser.add_argument('-k', '--keyFile', type=str, required=True, nargs='?', action='store', help='Key file (.csv)')
        self.parser.add_argument('-d', '--datasets', type=str, required=True, nargs='?', action='store', help='A path to a folder containing all the datasets (.csv)')
        self.parser.add_argument('-kt', '--keyTarg', required=True, nargs='+', action='store', help='a list of keyFile header strings that contains the 1) compoundClass 2) rowIDs')
        self.parser.add_argument('-e', '--exclude', required=False, default=None, action='store', help='an lst file (no header) containing a list of strings that define which classes to exclude')
        self.parser.add_argument('-o', '--outName', required=False, type=str, default='out', nargs='?', action='store', help='name for outputs')
        self.parser.add_argument('-n', '--name', required=False, type=FileType('r'), action='store', default=None, help='2-column lst file mapping dataset file name to desired name')
        self.parser.add_argument('-i', '--image', action='store_false', default=True, help='disables ecdf plotting (only generates p-value csv files)')
        self.parser.add_argument('-rp','--read-pickle', action='store_true', default=False, help='use precomputed (and pickled) nxn similarity matrix pd.DataFrame instead of nxm DataFrame (found inside the pickles directory inside the dataset directory)\n'+\
            "Also ignores '--name' parameter if provided")
        self.parser.add_argument('-p','--pickle', action='store_true', default=False, help = 'pickle save the computed nxn similarity matrices in the datasets path (will generated pickles folder in dataset directory)')

        #args
        if inOpts is None:
            self.args = self.parser.parse_args()
        else:
            self.args = self.parser.parse_args(inOpts)

def main(inOpts = None):
    cl = CommandLine(inOpts=inOpts)

    keyFile = cl.args.keyFile
    dsFolder = cl.args.datasets
    keyTargs = cl.args.keyTarg
    outFile = cl.args.outName
    prodImage = cl.args.image
    if cl.args.exclude is not None:
        excludes = [x.strip() for x in open(cl.args.exclude,'r').readlines()]
    else: excludes = cl.args.exclude
    if cl.args.name is not None:
        dsNames = {k:v for k,v in [x.strip().split('\t') for x in open(cl.args.name,'r').readlines()]}
    else: dsNames = cl.args.name
    
    key = pd.read_csv(keyFile, sep=',')

    ks = KSProcessingCLI(key=key, dataFolder=dsFolder, keyTargs=keyTargs, exclude=excludes, dsNames=dsNames, read_pickle=cl.args.read_pickle, pickle=cl.args.pickle)
    
    pdf = f'{outFile}ecdfGraphs.pdf'
    ks.plotCalcData(outName=pdf, produceImg=prodImage)
    pearson, euc = ks.generatePVals()

    pearson.to_csv(f'{outFile}Pearson.csv', sep =',')
    euc.to_csv(f'{outFile}Euc.csv', sep =',')

    ks.plotPVals(pearson=pearson, euc=euc, outName=outFile, width=1080, height=720)

if __name__ == '__main__':
    main()
