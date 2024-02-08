import os
import re
import io
import pathlib
import sys

from tqdm.auto import tqdm
import numpy as np
import modin.pandas as pd

from scipy.stats import kstest as kstest
from scipy.spatial.distance import pdist, squareform

from ks import *

def ensure_path_exists(file_path: pathlib.Path) -> pathlib.Path:
    # Convert the input to a Path object
    path = pathlib.Path(file_path)
   
    # Check if the directory of the path exists, and if not, create it
    path.mkdir(parents=True, exist_ok=True)

    return path

class KSProcessingPreProcess:
    def __init__(self, dataFolder: str, dsNames: dict = None, key: pd.DataFrame = None, keyTargs: list = None, exclude: list = None):
        
        #initializing data
        self.key = key
        self.dataFolder = dataFolder
        self.exclude = exclude
        self.keyTargs = keyTargs
        self.data = {}
        
        picklePath = ensure_path_exists(os.path.join(self.dataFolder,"hd5-pickles"))
        
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
                
                generateCorr(self.data[dictname]).\
                    to_hdf(picklePath / f"{dictname}_CorrMat.hd5",
                           key='corrDF',
                           format='table',
                           complevel=9,
                           mode='w')
                    
                generateEucDist(self.data[dictname]).\
                    to_hdf(picklePath / f"{dictname}_DistMat.hd5",
                           key='distDF',
                           format='table',
                           complevel=9,
                           mode='w') 
                
        else:
            for name, file in tqdm(dsNames.items()):
                currDataset = os.path.join(self.dataFolder, file)
                print(f"Reading dataset {file} as {name}",file=sys.stderr)
                self.data[name] = pd.read_csv(currDataset, index_col=0)
                if self.data[name].index.name is None:
                    self.data[name].index.name = 'Index'
                self.data[name] = dup(self.data[name])
                
                generateCorr(self.data[name]).\
                    to_hdf(os.path.join(picklePath , f"{name}_CorrMat.hd5"),
                           key='corrDF',
                           format='table',
                           complevel=9,
                           mode='w')
                    
                generateEucDist(self.data[name]).\
                    to_hdf(os.path.join(picklePath , f"{name}_DistMat.hd5"),
                           key='distDF',
                           format='table',
                           complevel=9,
                           mode='w') 
        
class CommandLine:
    def __init__(self, inOpts = None):
        from argparse import ArgumentParser

        self.parser = ArgumentParser(
            description='CLI preProcessor for KSTestApp program',
            prefix_chars='-',
            add_help=True,
            usage='python3 ksSimMat.py -d <FOLDER containing datasets> 
        )

        #arguments
        self.parser.add_argument('-d', '--datasets', type=str, required=True, nargs='?', action='store', help='A path to a folder containing all the datasets (.csv)')
        self.parser.add_argument('-n', '--name', required=False, type=argparse.FileType('r'), action='store', default=None, help='2-column lst file mapping dataset file name to desired name')
        self.parser.add_argument('-k', '--keyFile', type=str, required=False, nargs='?', action='store', help='Key file (.csv)')
        self.parser.add_argument('-kt', '--keyTarg', required=False, nargs='+', action='store', help='a list of keyFile header strings that contains the 1) compoundClass 2) rowIDs')
        self.parser.add_argument('-e', '--exclude', required=False, default=None, action='store', help='an lst file (no header) containing a list of strings that define which classes to exclude')
        # self.parser.add_argument('-o', '--outName', required=False, type=str, default='out', nargs='?', action='store', help='name for outputs')
        # self.parser.add_argument('-i', '--image', action='store_false', default=True, help='disables ecdf plotting (only generates p-value csv files)')
        # self.parser.add_argument('-rp','--read-pickle', action='store_true', default=False, help='use precomputed (and pickled) nxn similarity matrix pd.DataFrame instead of nxm DataFrame (found inside the pickles directory inside the dataset directory)\n'+\
        #     "Also ignores '--name' parameter if provided")
        # self.parser.add_argument('-p','--pickle', action='store_true', default=False, help = 'pickle save the computed nxn similarity matrices in the datasets path (will generated pickles folder in dataset directory)')

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
    if cl.arg.exclude is not None:
        excludes = [x.strip() for x in open(cl.args.exclude,'r').readlines()]
    if cl.args.name is not None:
        dsNames = {k,v for k,v in [x.strip().split('\t') for x in open(cl.args.name,'r').readlines()]}

    ks = KSProcessingCLI(key=key, dataFolder=dsFolder, keyTargs=keyTargs, exclude=excludes, dsNames=cl.args.name)
    
    pdf = f'{outFile}ecdfGraphs.pdf'
    ks.plotCalcData(outName=pdf, produceImg=prodImage)
    pearson, euc = ks.generatePVals()

    pearson.to_csv(f'{outFile}Pearson.csv', sep =',')
    euc.to_csv(f'{outFile}Euc.csv', sep =',')

    ks.plotPVals(pearson=pearson, euc=euc, outName=outFile, width=1080, height=720)

if __name__ == '__main__':
    main()

        
            