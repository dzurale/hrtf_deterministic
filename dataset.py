import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import h5py
from utils import get_anthro_norm_params, get_log_fft
from scipy.fft import fft, ifft

class HRTFDataset(Dataset):
    def __init__(self, dataPath, 
                 whichHRIR="trunc64",  
                 trainValOp="train", 
                 valSubjs=[], 
                 anthroFlag=False, 
                 anthroOp=None,
                 azOps=[6, 0],
                 elOps=[12, 0],
                 samplingOp="sym",
                 quadSelect="full"):
        """
        Inputs:
            dataPath:   Path to the hdf5 dataset
            
            whichHRIR:  Set "trunc64" for getting the 64 points itd and ild free hrirs
                        Set "raw" for getting the unprocessed 200 points hrirs
            
            trainValOp: Set "train" to get training data
                        Set "val" to get validation data
                         
            valSubjs:   List of subjects to be used for validation
            
            anthroFlag: Set True to only have subjects with anthro data
                        Set False to have all subjects with and without anthro data
                         
            anthroOp:   Set 8 for getting using the correlation studied 8 features
                        Set 27 to use all 27 features
                        
            azOps:      azOps[0] is the total number of points to be sampled in azimuth
                        azOps[1] is the point of first sampling point in azimuth
                        
            elOps:      elOps[0] is the total number of points to be sampled in elevation
                        elOps[1] is the point of first sampling point in elevation
            
            samplingOp: Set "sym" for symmetric sampling of azimuths and elevations about the centers
                        Set "uni" for uniform sampling of azimuths and elevations
                        
            quadSelect: This is the region in the sphere to select
                        Set "full" to sample throughout the sphere
                        Set "upper" to sample only in the upper half
        """    
        self.data = []
        hdf5File = h5py.File(dataPath, 'r')       
        
        if anthroOp is not None:
            muD, muX, muTheta, stdD, stdX, stdTheta = get_anthro_norm_params(hdf5File)
        
        if anthroFlag:
            validDataCondition = 4
        else:
            validDataCondition = 1
            
        if whichHRIR == "trunc64":
            hrirPathL = "/hrir_l/trunc_64"
            hrirPathR = "/hrir_r/trunc_64"
            nfft = 64
            hrirLength = 64
        elif whichHRIR == "raw":
            hrirPathL = "/hrir_l/raw"
            hrirPathR = "/hrir_l/raw"
            nfft = 256
            hrirLength = 200
        else:
            print("Invalid whichHRIR")
            
        posPath = "/srcpos/raw"
        
        azNofSamples = azOps[0]
        azStart = azOps[1]
        elNofSamples = elOps[0]
        elStart = elOps[1]
        
        if quadSelect == "full":
            azHopSize = int(np.floor(25/azNofSamples))
            elHopSize = int(np.floor(50/elNofSamples))
            
            if azStart >= azHopSize:
                print("Warning: Invalid value for azOps")
            if elStart > elHopSize:
                print("Warning: Invalid value for elOps")            
        
        elif quadSelect == "upper":
            azHopSize = int(np.floor(25/azNofSamples))
            elHopSize = int(np.float(33/elNofSamples))
        
            if azStart >= azHopSize:
                print("Warning: Invalid value for azOps")
            if elStart < 8 or elStart > 40 or elStart > elHopSize+8:
                print("Warning: Invalid value for elOps")
        
        else:
            print("Warning: Invalid quadSelect selection")
                
        
        if samplingOp == "sym":
            azEnd = int((azNofSamples/2 - 1) * azHopSize) + azStart
            elEnd = int((elNofSamples/2 - 1) * elHopSize) + elStart

            azIndices = np.concatenate((np.arange(azStart, azEnd + 1, azHopSize), 
                                        np.arange(24 - azEnd, 25 - azStart, azHopSize)))
            elIndices = np.concatenate((np.arange(elStart, elEnd + 1, elHopSize), 
                                        np.arange(49 - elEnd, 50 - elStart, elHopSize)))
        
        elif samplingOp == "uni":
            azEnd = (azNofSamples-1)*azHopSize + azStart
            elEnd = (elNofSamples-1)*elHopSize + elStart

            azIndices = np.arange(azStart, azEnd + 1, azHopSize)
            elIndices = np.arange(elStart, elEnd + 1, elHopSize)
            
        else:
            print("Warning: Invalid samplingOp selection")
                    
        dataIdx = 0
        testIdx = 0
        for idx, subjects in enumerate(hdf5File):
            if len(hdf5File[subjects]) > validDataCondition:
                    
                # Left ear:
                # Read file
                hrirL = np.array(hdf5File.get(subjects+hrirPathL))
                
                # Process HRTFs:                
                hrirL = hrirL / 2
                hrtfL = get_log_fft(hrirL, nfft=nfft).T
                hrtfL = hrtfL / 6 + 1                
                hrtfL = np.reshape(hrtfL, [int(nfft/2), 25, 50])
                hrtfSparseL = np.take(np.take(hrtfL, azIndices, 1), elIndices, 2)
                    
                # Process HRIRs:
                hrirL = (hrirL + 1) / 2
                hrirL = hrirL.T
                hrirL = np.reshape(hrirL, [hrirLength, 25, 50])
                hrirSparseL = np.take(np.take(hrirL, azIndices, 1), elIndices, 2)
                
                
                # Right ear:
                # Read file
                hrirR = np.array(hdf5File.get(subjects+hrirPathR))
                
                # Process HRTFs:
                hrirR = hrirR / 2
                hrtfR = get_log_fft(hrirR, nfft=nfft).T
                hrtfR = hrtfR / 6 + 1                
                hrtfR = np.reshape(hrtfR, [int(nfft/2), 25, 50])
                hrtfSparseR = np.take(np.take(hrtfR, azIndices, 1), elIndices, 2)
                
                # Process HRIRs:                
                hrirR = (hrirR + 1) / 2
                hrirR = hrirR.T
                hrirR = np.reshape(hrirR, [hrirLength, 25, 50])
                hrirSparseR = np.take(np.take(hrirR, azIndices, 1), elIndices, 2)
                
                
                # Reshaping the reference position vector to match the HRIR reshaping
                posArray = np.array(hdf5File.get(subjects+posPath))
                posArray = posArray.T
                posArray = np.reshape(posArray, [3, 25, 50])
                
                if anthroOp is not None:
                    # ANTHROPOMETRY L&R:
                    thisSubj = hdf5File[subjects]
                    D = thisSubj.attrs['D']
                    X = thisSubj.attrs['X']
                    theta = thisSubj.attrs['theta']

                    normD = 1/(1+np.exp(-(D - muD)/stdD))
                    normX = 1/(1+np.exp(-(X - muX)/stdX))
                    normTheta = 1/(1+np.exp(-(theta - muTheta)/stdTheta))

                    normD_L = D[:8]
                    normD_R = D[8:]
                    normTheta_L = theta[:2]
                    normTheta_R = theta[2:]

                    if anthroOp == 8:
                        anthroL = np.concatenate((np.take(normX, [2, 4]), np.take(normD_L, [2, 3, 4, 5]), normTheta_L))
                        anthroR = np.concatenate((np.take(normX, [2, 4]), np.take(normD_R, [2, 3, 4, 5]), normTheta_R))
                    elif anthroOp == 27:
                        anthroL = np.concatenate((normX, normD_L, normTheta_L))
                        anthroR = np.concatenate((normX, normD_R, normTheta_R))
                
                if trainValOp == "train":
                    if testIdx < len(valSubjs):
                        if valSubjs[testIdx] == dataIdx:
                            dataIdx+=1
                            testIdx+=1
                            continue
                                                        
                    features = dict()

                    features["hrtf_l"] = torch.from_numpy(hrtfL).float()
                    features["hrtf_r"] = torch.from_numpy(hrtfR).float()
                    
                    features["sparse_hrtf_l"] = torch.from_numpy(hrtfSparseL).float()
                    features["sparse_hrtf_r"] = torch.from_numpy(hrtfSparseR).float()

                    features["hrir_l"] = torch.from_numpy(hrirL).float()
                    features["hrir_r"] = torch.from_numpy(hrirR).float()

                    features["sparse_hrir_l"] = torch.from_numpy(hrirSparseL).float()
                    features["sparse_hrir_r"] = torch.from_numpy(hrirSparseR).float()

                    features["pos_array"] = torch.from_numpy(posArray).float()
                    
                    if anthroOp is not None:
                        features["anthro_l"] = torch.from_numpy(anthroL).float()
                        features["anthro_r"] = torch.from_numpy(anthroR).float()

                    self.data.append(features)           
                            
                elif trainValOp == "val":
                    if testIdx < len(valSubjs):
                        if valSubjs[testIdx] == dataIdx:
                            features = dict()

                            features["hrtf_l"] = torch.from_numpy(hrtfL).float()
                            features["hrtf_r"] = torch.from_numpy(hrtfR).float()
                            
                            features["sparse_hrtf_l"] = torch.from_numpy(hrtfSparseL).float()
                            features["sparse_hrtf_r"] = torch.from_numpy(hrtfSparseR).float()

                            features["hrir_l"] = torch.from_numpy(hrirL).float()
                            features["hrir_r"] = torch.from_numpy(hrirR).float()
                            
                            features["sparse_hrir_l"] = torch.from_numpy(hrirSparseL).float()
                            features["sparse_hrir_r"] = torch.from_numpy(hrirSparseR).float()

                            features["pos_array"] = torch.from_numpy(posArray).float()
                            
                            if anthroOp is not None:
                                features["anthro_l"] = torch.from_numpy(anthroL).float()
                                features["anthro_r"] = torch.from_numpy(anthroR).float()

                            self.data.append(features)
                            
                            testIdx+=1
                    else:
                        break
                    
            dataIdx+=1                
                        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    

def get_dataloaders(dataset, batchSize=1, shuffle=True):
    trainDL = DataLoader(dataset, batch_size=batchSize, shuffle=shuffle)
    
    return trainDL

if __name__ == "__main__":
    pass