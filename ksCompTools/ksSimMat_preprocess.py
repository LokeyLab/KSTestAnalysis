import os
import re
import io
import pathlib
import sys
import itertools

from tqdm.auto import tqdm
from concurrent import futures
import numpy as np
# import modin.pandas as pd
from ks import *
import pandas as pd

from scipy.stats import kstest as kstest
from scipy.spatial.distance import pdist, squareform


def ensure_path_exists(file_path: pathlib.Path) -> pathlib.Path:
    # Convert the input to a Path object
    path = pathlib.Path(file_path)
   
    # Check if the directory of the path exists, and if not, create it
    path.mkdir(parents=True, exist_ok=True)

    return path

def findNaNs (simMat: pd.DataFrame = None,simMat_path: str =None) -> dict:
    nanOffenders = dict()
    if simMat is None and simMat_path is not None:
        simMat = pd.read_pickle(simMat_path)
    elif simMat is not None and simMat_path is None:
        simMat = simMat
    print("Full simMat Shape:", simMat.shape,file=sys.stderr)
    print("SimMat dropna Shape:", simMat.dropna().shape,file=sys.stderr)
    naCounts = simMat.isna().sum(axis=1)
    afflicted = simMat.loc[simMat.isna().sum(axis=1)>0].index.to_list()
    print("number of afflicted nan columns:",len(afflicted),file=sys.stderr)
    for aff in afflicted:
        for i in simMat[aff].isna().argsort()[::-1][:naCounts[aff]].index:
            # print(i)
            if i not in nanOffenders:
                nanOffenders[i] =1
            elif i in nanOffenders:
                nanOffenders[i] +=1
    nanOffs = pd.Series(nanOffenders)
    nanOffs = naCounts.index[(pd.Series(nanOffs) - naCounts)>0].to_list()
    nanOffenders = {k:v for k,v in nanOffenders.items() if k in nanOffs}
    return nanOffenders

class KSProcessingPreProcess:
    def __init__(self, dataFolder: str, dsNames: dict = None, key: pd.DataFrame = None, keyTargs: list = None, exclude:list = None, include: list = None, min_group_size: int = 3,parallel:bool = False):
        
        #initializing data
        self.key = key
        self.dataFolder = dataFolder
        self.exclude = exclude
        self.include = include
        self.keyTargs = keyTargs
        self.dsNames = dsNames
        self.data = {}
        
        if self.key is not None and self.keyTargs is not None:
            # determine groups
            self.groups = self.key[self.keyTargs[0]].value_counts(dropna=True, ascending=False)
            self.groups = self.groups[self.groups > min_group_size]
            if self.include is not None and self.exclude is None:
                self.groups = self.groups[self.groups.index.isin(self.include)]
            elif self.exclude is not None and self.include is None:
                self.groups = self.groups[~self.groups.index.isin(self.exclude)]
            elif self.exclude is None and self.include is None:
                pass
            else:
                print(f"INCLUDE AND EXCLUDE OPTIONS ARE MUTUALLY EXCLUSIVE.\n PROVIDE ON OR THE OTHER, NOT BOTH.",file=sys.stderr)
            self.groups = list(self.groups.index)

            #get final compounds and group ids
            self.comps = self.key[[*self.keyTargs]]
            self.comps = self.comps[self.comps.iloc[:,0].isin(self.groups)]
            self.comps = self.comps.groupby(self.comps.iloc[:,0])[self.comps.columns[1]]
            self.comps = {k: np.array(v) for k,v in self.comps}
            print({k:len(v) for k,v in self.comps.items()},file=sys.stderr)
            self.comps_expanded = list()
            [self.comps_expanded.extend(c) for c in self.comps.values()]
        self.picklePath = ensure_path_exists(os.path.join(self.dataFolder,"pickles"))
        
        if parallel:
            exec_list = list()
            if self.dsNames is None:
                print(f"No dsNames provided, using filenames: {glob.glob(self.dataFolder+'/*.csv')}",file=sys.stderr)
                for f in tqdm(glob.glob(self.dataFolder+"/*.csv")):
                    currDataset = os.path.join(self.dataFolder, os.path.basename(file))
                    dictname = os.path.basename(currDataset).replace('.csv','')
                    exec_list.append((dictname,f))
            else:
                print(f"dsNames provided. they are: {self.dsNames}",file=sys.stderr)
                for name, f in tqdm(self.dsNames.items()):
                    exec_list.append((name,f))
            self._exec_parallel(exec_list)
        
        else:
            self._single_thread()
    
    def _single_thread(self):
        
        print(f"reading datafiles from {self.dataFolder}",file=sys.stderr)
        # parse data Folder
        if self.dsNames is None:
            print(f"No dsNames provided, using filenames: {glob.glob(self.dataFolder+'/*.csv')}",file=sys.stderr)
            for f in tqdm(glob.glob(self.dataFolder+"/*.csv")):
                currDataset = os.path.join(self.dataFolder, os.path.basename(file))
                dictname = os.path.basename(currDataset).replace('.csv','')
                print(f"reading dataset {dictname}",file=sys.stderr)
                self.data[dictname] = pd.read_csv(currDataset, index_col=0)
                if self.data[dictname].index.name is None:
                    self.data[dictname].index.name = 'Index'
                print(f"PRE DEDUP DATA SHAPE: {self.data[dictname].shape}",file=sys.stderr)
                self.data[dictname] = dup(self.data[dictname])
                print(f"FINISHED READING AND DUP ON {dictname}, dataset shape is {self.data[dictname].shape}",file=sys.stderr)
                # self.data[dictname].dropna(axis='index',how='any',inplace=True)
                
                if self.comps:
                    print(f"Prefiltering index with key {self.key}.",file=sys.stderr)
                    print(f"Keeping target groups: {sorted(self.groups)}",file=sys.stderr)
                    # clear datasets of rows not present in the key
                    # comps_expanded = list()
                    # [comps_expanded.extend(c) for c in self.comps.values()]
                    self.data[dictname] = self.data[dictname].loc[\
                        self.data[dictname].index.isin(\
                            comps_expanded)].copy()
                
                print(f"CALCULATING SIMILARITY MATRIX FOR {dictname}, will purge any pairwise  with nan") 
                cor = generateCorr(self.data[dictname])
                cor.drop(index=findNaNs(cor).keys(),inplace=True)
                cor.to_pickle(\
                    os.path.join(self.picklePath ,f"{dictname}_CorrMat.pkl.gz"),
                              compression={'method':'gzip','compresslevel':9}
                              )
                    
                euc = generateEucDist(self.data[dictname])
                euc.drop(index=findNaNs(euc).keys(),inplace=True)
                euc.to_pickle(\
                    os.path.join(self.picklePath , f"{dictname}_DistMat.pkl.gz"),
                    compression={'method':'gzip','compresslevel':9}
                    )
                print(f"Finished calculating and Pickling dataset: {dictname} with dimesions: {self.data[dictname].shape}",file=sys.stderr)          
        else:
            print(f"dsNames provided. they are: {self.dsNames}",file=sys.stderr)
            for name, f in tqdm(dsNames.items()):
                currDataset = os.path.join(self.dataFolder, f)
                print(f"Reading dataset {f} as {name}",file=sys.stderr)
                self.data[name] = pd.read_csv(currDataset, index_col=0)
                if self.data[name].index.name is None:
                    self.data[name].index.name = 'Index'
                print(f"PRE DEDUP DATA SHAPE: {self.data[name].shape}",file=sys.stderr)
                self.data[name] = dup(self.data[name])
                print(f'FINISHED READING AND DUP ON {name}, dataset shape is {self.data[name].shape}',file=sys.stderr)
                
                if self.comps:
                    print(f"Prefiltering index with key \n{key}.",file=sys.stderr)
                    print(f"Keeping target groups: {sorted(self.groups)}",file=sys.stderr)
                    # clear datasets of rows not present in the key
                    # print(self.data[name].shape,
                        #   self.data[name].index,
                        #   len(comps_expanded),
                        #   self.key.shape, len(self.comps))
                    self.data[name] = self.data[name].loc[\
                        self.data[name].index.isin(\
                            self.comps_expanded)].copy()
                    # sys.exit(0)
                generateCorr(self.data[name]).to_pickle(
                    os.path.join(self.picklePath ,f"{name}_CorrMat.pkl.gz"),
                              compression={'method':'gzip','compresslevel':9}
                              )
                    
                generateEucDist(self.data[name]).to_pickle(
                    os.path.join(self.picklePath , f"{name}_DistMat.pkl.gz"),
                    compression={'method':'gzip','compresslevel':9}
                    )
                print(f"Finished calculating and Pickling dataset: {name} with dimesions: {self.data[name].shape}",file=sys.stderr)
                
        print("FINISHED READING FROM FILE: length of datasets collected:",len(self.data),file=sys.stderr)
    
    def _parallel(self,name,f):
        currDataset = os.path.join(self.dataFolder, f)
        print(f"Reading dataset {f} as {name}",file=sys.stderr)
        self.data[name] = pd.read_csv(currDataset, index_col=0)
        if self.data[name].index.name is None:
            self.data[name].index.name = 'Index'
        print(f"PRE DEDUP DATA SHAPE: {self.data[name].shape}",file=sys.stderr)
        self.data[name] = dup(self.data[name])
        print(f'FINISHED READING AND DUP ON {name}, dataset shape is {self.data[name].shape}',file=sys.stderr)
        
        if self.comps:
            print(f"Prefiltering index with key \n{key}.",file=sys.stderr)
            print(f"Keeping target groups: {sorted(self.groups)}",file=sys.stderr)
            # clear datasets of rows not present in the key
            # print(self.data[name].shape,
                #   self.data[name].index,
                #   len(comps_expanded),
                #   self.key.shape, len(self.comps))
            self.data[name] = self.data[name].loc[\
                self.data[name].index.isin(\
                    self.comps_expanded)].copy()
            # sys.exit(0)
        generateCorr(self.data[name]).to_pickle(
            os.path.join(picklePath ,f"{name}_CorrMat.pkl.gz"),
                        compression={'method':'gzip','compresslevel':9}
                        )
            
        generateEucDist(self.data[name]).to_pickle(
            os.path.join(picklePath , f"{name}_DistMat.pkl.gz"),
            compression={'method':'gzip','compresslevel':9}
            )
        print(f"Finished calculating and Pickling dataset: {name} with dimesions: {self.data[name].shape}",file=sys.stderr)
        
    def _exec_parallel(argslist: list[tuple[str, str]],
                     stop_on_exception: bool = False,
                     max_workers:int = 4) -> dict[str, Exception]:
        errors = {}
        with futures.ProcessPoolExecutor(max_workers=max_workers
                                        ) as executor:
            # associate each future with its arguments, so we can go back and check which images failed (if any)
            submitted = {executor.submit(_parallel, *args): args for args in argslist}
            try:
                for future in tqdm(futures.as_completed(submitted), total=len(submitted)):
                    if future.cancelled():  # handle the case where the future was canceled by the code below if an exception was raised by a different future...
                        continue
                    name,f = submitted[future]
                    if (exception := future.exception()) is not None:
                        errors[image_template] = exception
                        if stop_on_exception:
                            # cancel all; it's a no-op if the future is already finished, so can do this en masse
                            for future in submitted:
                                future.cancel()
                            stop_on_exception = False  # don't need to re-cancel everything if we see further exceptions
            except BaseException:  # we want to catch everything, including KeyboardInterrupt, SystemExit, etc.
                for future in submitted:
                    future.cancel()
                raise
        return errors

    
                
class CommandLine:
    def __init__(self, inOpts = None):
        from argparse import ArgumentParser, FileType

        self.parser = ArgumentParser(
            description='CLI preProcessor for KSTestApp program',
            prefix_chars='-',
            add_help=True,
            usage='python3 ksSimMat.py -k <.csv file> -d <FOLDER containing datasets> -kt <list of header strings of key columns> -e <list file of classes to exclude> -n <name translation file of datasets>' 
        )

        #arguments
        self.parser.add_argument('-d', '--datasets', type=str, required=True, nargs='?', action='store', help='A path to a folder containing all the datasets (.csv)')
        self.parser.add_argument('-n', '--name', required=False, type=FileType('r'), action='store', default=None, help='2-column lst file mapping dataset file name to desired name')
        self.parser.add_argument('-k', '--keyFile', type=str, required=False, nargs='?', action='store', help='Key file (.csv)')
        self.parser.add_argument('-kt', '--keyTarg', required=False, nargs='+', action='store', help='a list of keyFile header strings that contains the 1) compoundClass 2) rowIDs')
        self.filterGroup = self.parser.add_mutually_exclusive_group()
        self.filterGroup.add_argument('-e', '--exclude', required=False, default=None, action='store', help='an lst file (no header) containing a list of strings that define which classes to exclude')
        self.filterGroup.add_argument('-in', '--include', required=False, default=None, action='store', help='an lst file (no header) containing a list of strings that define which classes to include')
        self.parser.add_argument('-s', '--min_group_size', required=False, default=3, action='store',type=int, help='Minimum number of representatives in the key for each group. Default: 3')
        self.parser.add_argument('--parallel', required=False, default=False, action='store_true', help='execute calculation in parallel')
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
    if cl.args.exclude is not None:
        excludes = [x.strip() for x in open(cl.args.exclude,'r').readlines()]
    else: excludes = cl.args.exclude
    if cl.args.include is not None:
        includes = [x.strip() for x in open(cl.args.include,'r').readlines()]
    else: includes = cl.args.include
    if cl.args.name is not None:
        dsNames = {k:v for k,v in [x.strip().split('\t') for x in cl.args.name.readlines()]}
    else: dsNames = cl.args.name
        
    key = pd.read_csv(keyFile, sep=',')
    print(excludes, dsNames,cl.args)
    # exit(0)
    ks = KSProcessingPreProcess(key=key, dataFolder=dsFolder, keyTargs=keyTargs, exclude=excludes, dsNames=dsNames,include=includes,min_group_size=cl.args.min_group_size,parallel=cl.args.parallel)
    

if __name__ == '__main__':
    main()

        
            