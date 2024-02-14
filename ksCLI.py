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

def wideDf_to_hdf(filename, data, columns=None, maxColSize=2000, **kwargs):
    """Write a `pandas.DataFrame` with a large number of columns
    to one HDFStore. From: https://stackoverflow.com/a/51931059

    Parameters
    -----------
    filename : str
        name of the HDFStore
    data : pandas.DataFrame
        data to save in the HDFStore
    columns: list
        a list of columns for storing. If set to `None`, all 
        columns are saved.
    maxColSize : int (default=2000)
        this number defines the maximum possible column size of 
        a table in the HDFStore.

    """
    import numpy as np
    from collections import ChainMap
    store = pd.HDFStore(filename, **kwargs)
    if columns is None:
        columns = data.columns
    colSize = columns.shape[0]
    if colSize > maxColSize:
        numOfSplits = np.ceil(colSize / maxColSize).astype(int)
        colsSplit = [
            columns[i * maxColSize:(i + 1) * maxColSize]
            for i in range(numOfSplits)
        ]
        _colsTabNum = ChainMap(*[
            dict(zip(columns, ['data{}'.format(num)] * colSize))
            for num, columns in enumerate(colsSplit)
        ])
        colsTabNum = pd.Series(dict(_colsTabNum)).sort_index()
        for num, cols in enumerate(colsSplit):
            store.put('data{}'.format(num), data[cols], format='table')
        store.put('colsTabNum', colsTabNum, format='fixed')
    else:
        store.put('data', data[columns], format='table')
    store.close()

def read_hdf_wideDf(filename, columns=None, **kwargs):
    """Read a `pandas.DataFrame` from a HDFStore.
    From: https://stackoverflow.com/a/51931059

    Parameter
    ---------
    filename : str
        name of the HDFStore
    columns : list
        the columns in this list are loaded. Load all columns, 
        if set to `None`.

    Returns
    -------
    data : pandas.DataFrame
        loaded data.

    """
    store = pd.HDFStore(filename)
    data = []
    colsTabNum = store.select('colsTabNum')
    if colsTabNum is not None:
        if columns is not None:
            tabNums = pd.Series(
                index=colsTabNum[columns].values,
                data=colsTabNum[columns].data).sort_index()
            for table in tabNums.unique():
                data.append(
                    store.select(table, columns=tabsNum[table], **kwargs))
        else:
            for table in colsTabNum.unique():
                data.append(store.select(table, **kwargs))
        data = pd.concat(data, axis=1).sort_index(axis=1)
    else:
        data = store.select('data', columns=columns)
    store.close()
    return data


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

        #print("CompoundClass to compoundID groups:",self.comps,file=sys.stderr)
        #print("inputs are:",read_pickle,dsNames, self.data,self.exclude,self.dataFolder,file=sys.stderr)
        if not read_pickle:
            print(f"reading datafile from {self.dataFolder}",file=sys.stderr)
            # parse data Folder
            if dsNames is None:
                print(f"No dsNames provided, using filenames: {glob.glob(self.dataFolder+'/*.csv')}",file=sys.stderr)
                for f in tqdm(glob.glob(self.dataFolder+"/*.csv")):
                    currDataset = os.path.join(self.dataFolder, os.path.basename(f))
                    dictname = os.path.basename(currDataset).replace('.csv','')
                    print(f"reading dataset {dictname}",file=sys.stderr)
                    self.data[dictname] = pd.read_csv(currDataset, index_col=0)
                    if self.data[dictname].index.name is None:
                        self.data[dictname].index.name = 'Index'
                    print(f"PRE DEDUP DATA SHAPE: {self.data[dictname].shape}",file=sys.stderr)
                    self.data[dictname] = dup(self.data[dictname])
                    df = self.data[dictname]
                    print(f"FINISHED READING AND DUP ON {dictname}, dataset shape is {df.shape}",file=sys.stderr)
                    # clear datasets of rows not present in the key
                    self.data[dictname] = df.loc[df.index.isin(self.key[self.keyTargs[1]].to_list())].copy()
                    del df
            else:
                print(f"dsNames provided. they are: {dsNames}",file=sys.stderr)
                for name, f in tqdm(dsNames.items()):
                    #, glob.glob(self.dataFolder+"/*.csv"))):
                    currDataset = os.path.join(self.dataFolder,f)
                    print(f"Reading dataset {f} as {name}",file=sys.stderr)
                    self.data[name] = pd.read_csv(currDataset, index_col=0)
                    if self.data[name].index.name is None:
                        self.data[name].index.name = 'Index'
                    print(f"PRE DEDUP DATA SHAPE: {self.data[name].shape}",file=sys.stderr)
                    self.data[name] = dup(self.data[name])
                    df = self.data[name]
                    print(f'FINISHED READING AND DUP ON {name}, dataset shape is {df.shape}',file=sys.stderr)
                    # clear datasets of rows not present in the key
                    self.data[name] = df.loc[df.index.isin(self.key[self.keyTargs[1]].to_list())].copy()
                    del df
            print("FINISHED READING FROM FILE: length of datasets collected:",len(self.data),file=sys.stderr)        
            self.datasetCorr = {k:generateCorr(v) for k,v in self.data.items()}
            self.datasetDist = {k:generateEucDist(v) for k,v in self.data.items()}
            
            # once the nxn correlation and distmats are computed
            # OG dfs not needed, keep the df_names (keys)
            #self.data = self.data.keys()
            
        else:
            print(f"reading pickled simMats from {os.path.join(self.dataFolder,'pickles')}",file=sys.stderr)
            # read pickled simMats from pickle folder inside data folder
            # dsNames ignored, only use pickle names
            self.datasetCorr = dict()
            self.datasetDist = dict()
            for pickleFile in tqdm(glob.glob(self.dataFolder+"/pickles/*.hd5")):
                file = pickleFile.replace('.hd5','')
                if file.endswith('_CorrMat'):
                    self.datasetCorr[file.replace('_CorrMat','')] = read_hdf_wideDf(pickleFile)#,'corrDF')
                elif file.endswith('_DistMat'):
                    self.datasetDist[file.replace('_DistMat','')] = read_hdf_wideDf(pickleFile)#,'distDF')

        if pickle:
            picklePath = ensure_path_exists(os.path.join(self.dataFolder,"pickles"))
            #key='corrDF'
            # [wideDf_to_hdf(os.path.join(picklePath , f"{k}_CorrMat.hd5"),data=corrDF,format='table',complevel=9,mode='w') for k,corrDF in tqdm(self.datasetCorr.items())]
            [wideDf_to_hdf(os.path.join(picklePath , f"{k}_CorrMat.pkl.gz"),data=corrDF,format='table',complevel=9,mode='w') for k,corrDF in tqdm(self.datasetCorr.items())]
            #key='distDF'
            # [wideDf_to_hdf(os.path.join(picklePath , f"{k}_DistMat.hd5"),data=distDF,format='table', complevel=9,mode='w') for k,distDF in tqdm(self.datasetDist.items())]
            [distDF.to_pickle(os.path.join(picklePath , f"{k}_DistMat.pkl.gz"),data=distDF,format='table', complevel=9,mode='w') for k,distDF in tqdm(self.datasetDist.items())]
            
        assert len(self.datasetCorr) == len(self.datasetDist)

        # generate useable datasets
        self.names = [k for k in sorted(self.datasetCorr.keys())]
        self.datasetsCorrList = [v for k,v in sorted(self.datasetCorr.items(), key=lambda a:a[0])]
        self.datasetsDistList = [v for k,v in sorted(self.datasetDist.items(), key=lambda a:a[0])]

        print("n x n pairwise distmats calc'd. length of dataset lists:", len(self.names),len(self.datasetCorr),len(self.datasetDist),file=sys.stderr)

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
            usage='python3 ksCLI.py -k <.csv file> -d <FOLDER containing datasets> -kt <list of header strings of key columns> -e <list file of classes to exclude>'
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
    outFile = ensure_path_exists(cl.args.outName)
    prodImage = cl.args.image
    if cl.args.exclude is not None:
        excludes = [x.strip() for x in open(cl.args.exclude,'r').readlines()]
    else: excludes = cl.args.exclude
    if cl.args.name is not None:
        dsNames = {k:v for k,v in [x.strip().split('\t') for x in cl.args.name.readlines()]}
    else: dsNames = cl.args.name
    
    key = pd.read_csv(keyFile, sep=',')
    print(excludes, cl.args,file=sys.stderr)
    #exit(0)
    ks = KSProcessingCLI(key=key, dataFolder=dsFolder, keyTargs=keyTargs, exclude=excludes, dsNames=dsNames, read_pickle=cl.args.read_pickle, pickle=cl.args.pickle)
    
    pdf = f'{outFile}ecdfGraphs.pdf'
    ks.plotCalcData(outName=pdf, produceImg=prodImage)
    pearson, euc = ks.generatePVals()

    pearson.to_csv(f'{outFile}Pearson.csv', sep =',')
    euc.to_csv(f'{outFile}Euc.csv', sep =',')

    ks.plotPVals(pearson=pearson, euc=euc, outName=outFile, width=1080, height=720)

if __name__ == '__main__':
    main()
