import os
import torch
import torch.nn as nn
from models import AEEncoder, AEDecoder, weights_init
import numpy as np
import h5py
from dataset import HRTFDataset, get_dataloaders
import time
import global_params as gp
from utils import get_lsd, get_log_fft
from torch.utils.tensorboard import SummaryWriter

def train(encoder, 
          decoder,
          hrtfType,
          trainDL, 
          valDL,
          encLR,
          decLR,
          numEpochs,
          device):
    
    # Global parameters
    logFolder = gp.logFolder
    exp = gp.exp
    saveFlag = gp.saveFlag
    saveFolder = gp.saveFolder
    saveFile = gp.saveFile
    
    # Bookkeeping setup
    ts = time.strftime('%Y_%b_%d_%H_%M_%S', time.gmtime())
    writer = SummaryWriter(os.path.join(logFolder, repr(ts)+"_"+exp))
    
    # Initialize weights
    encoder.apply(weights_init)
    decoder.apply(weights_init)

    # Initialize optimizers
    encOptimizer = torch.optim.Adam(encoder.parameters(), lr=encLR, betas=(0.5, 0.999))
    decOptimizer = torch.optim.Adam(decoder.parameters(), lr=decLR, betas=(0.5, 0.999))
    
    def train_epoch():
        lossEp = 0
        for batchNo, batch in enumerate(trainDL):
            # Get batch
            if hrtfType is "hrtf":
                inL = batch["hrtf_l"].to(device)
                targetL = inL
            elif hrtfType == "hrir":
                inL = batch["hrir_l"].to(device)
                targetL = inL
                
            # Initializations
            # Clear gradients
            encOptimizer.zero_grad()
            decOptimizer.zero_grad()
                                        
            # Forward pass
            predL = decoder(encoder(inL))
            
            # Get loss
            loss = torch.mean(get_lsd(predL, targetL))
            
            # Calculate new grad
            loss.backward()
            
            # update weights
            encOptimizer.step()
            decOptimizer.step()
            
            lossEp+=loss.item()
            
        lossEp = lossEp/(batchNo+1)
        
        return lossEp
    
    def get_val_loss():
        encoder.eval()
        decoder.eval()
        lossEp = 0
        for batchNo, batch in enumerate(valDL):
            # Get batch
            if hrtfType is "hrtf":
                inL = batch["hrtf_l"].to(device)
                targetL = inL
            elif hrtfType == "hrir":
                inL = batch["hrir_l"].to(device)
                targetL = inL
                                        
            # Forward pass
            predL = decoder(encoder(inL))
            
            # Get loss
            loss = torch.mean(get_lsd(predL, targetL))
            
            lossEp+=loss.item()
            
        lossEp = lossEp/(batchNo+1)
        
        encoder.train()
        decoder.train()
        
        return lossEp
            
    # Main training loop
    print("Begin training")
    bestLoss = 10000
    bestEp = 0
    for thisEpoch in range(numEpochs):
        lossTrain = train_epoch()
        lossVal = get_val_loss()
        if hrtfType == "hrtf":
            lossTrain = lossTrain * 6 * 20
            lossVal = lossVal * 6 * 20

        print("Epoch %d :- Train LSD: %.5f, Val LSD: %.5f" % 
              (thisEpoch+1, lossTrain, lossVal)) 

        if lossVal < bestLoss:
            bestEp = thisEpoch + 1
            bestLoss = lossVal

            # To save every successive best model
            if saveFlag:
                torch.save({'encoder': encoder.state_dict(), 
                            'decoder': decoder.state_dict(),
                            'encOptimizer': encOptimizer.state_dict(), 
                            'decOptimizer': decOptimizer.state_dict()}, 
                           saveFolder+saveFile%numEpochs)

        writer.add_scalar("Training Loss LSD", lossTrain, thisEpoch)
        writer.add_scalar("Validation Loss LSD", lossVal, thisEpoch)

        if thisEpoch - bestEp > 10000:
            print("Best Epoch: %d, Best Val Loss: %.5f" % (bestEp, bestLoss))
            if saveFlag:
                os.rename(saveFolder+saveFile%numEpochs, saveFolder+saveFile%bestEp)
            break                 
                
    return encoder, decoder

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using %s" % device)
    
    # Get Data    
    datasetTrain = HRTFDataset(gp.dataPath, gp.whichHRIR, 
                               "train", gp.valSubjs, 
                               gp.anthroFlag, gp.anthroOp,
                               gp.azOps, gp.elOps, gp.samplingOp, gp.quadSelect)    
    trainDL = get_dataloaders(datasetTrain, batchSize=datasetTrain.__len__())
    
    # Get validation Data
    datasetVal = HRTFDataset(gp.dataPath, gp.whichHRIR, 
                             "val", gp.valSubjs, 
                             gp.anthroFlag, gp.anthroOp, 
                             gp.azOps, gp.elOps, gp.samplingOp, gp.quadSelect)
    valDL = get_dataloaders(datasetVal, batchSize=datasetVal.__len__())
    
    # Initialize models    
    encoder = AEEncoder(inChannels=gp.inOutChannels, dsType=gp.encDSType, BNFlag=gp.encBNFlag,
                        modelDepth=gp.modelDepth, featMapsMultFact=gp.featMapsMultFact, numConvBlocks=gp.numConvBlocks,
                        dropoutP=gp.dropoutP).to(device)
    decoder = AEDecoder(outChannels=gp.inOutChannels, finalActType=gp.decFinalActType, BNFlag=gp.decBNFlag, 
                        modelDepth=gp.modelDepth, featMapsMultFact=gp.featMapsMultFact, numConvBlocks=gp.numConvBlocks,
                        outPadArray=gp.decOutPadArray, dropoutP=gp.dropoutP).to(device)
    
    modelOut = train(encoder=encoder, 
                     decoder=decoder,
                     hrtfType=gp.hrtfType,
                     trainDL=trainDL, 
                     valDL=valDL,
                     encLR=gp.encLR,
                     decLR=gp.decLR,
                     numEpochs=gp.numEpochs,
                     device=device)