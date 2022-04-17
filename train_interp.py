import os
import torch
import torch.nn as nn
from models import AEEncoder, AEDecoder, SparseConvEncoder, SparseTConvEncoder, weights_init
import numpy as np
import h5py
from dataset import HRTFDataset, get_dataloaders
import time
import global_params as gp
from utils import get_lsd, get_log_fft
from torch.utils.tensorboard import SummaryWriter

def train(interpEncoder,
          aeDecoder, 
          hrtfType, 
          trainDL, 
          valDL, 
          encLR, 
          numEpochs, 
          decLR, 
          decFTEpochs,
          device):
    
    # Global parameters
    logFolder = gp.logFolder
    exp = gp.exp
    interpVer = gp.interpVer
    saveFlag = gp.saveFlag
    saveFolder = gp.saveFolder
    saveFile = gp.interpSaveFile
    
    hrtfType = gp.hrtfType
    
    decTrainMode = gp.interpDecTrainMode
    
    # Bookkeeping setup
    ts = time.strftime('%Y_%b_%d_%H_%M_%S', time.gmtime())
    writer = SummaryWriter(os.path.join(logFolder, repr(ts)+"_"+exp))
    
    # Initialize sparse encoder weights
    interpEncoder.apply(weights_init)
    if decTrainMode == "end2end":
        aeDecoder.apply(weights_init)

    # Initialize optimizers
    interpOptimizer = torch.optim.Adam(interpEncoder.parameters(), lr=encLR, betas=(0.5, 0.999))
    if decFTEpochs or (decTrainMode == "end2end"):
        decOptimizer = torch.optim.Adam(aeDecoder.parameters(), lr=decLR, betas=(0.5, 0.999))
        
    def encoder_train_epoch():
        lossEp = 0
        for batchNo, batch in enumerate(trainDL):
            # Get batch
            if hrtfType is "hrtf":
                targetL = batch["hrtf_l"].to(device)
                sparseL = batch["sparse_hrtf_l"].to(device)
            elif hrtfType == "hrir":
                targetL = batch["hrir_l"].to(device)
                sparseL = batch["sparse_hrir_l"].to(device)
                
            # Initializations
            # Clear gradients
            interpOptimizer.zero_grad()
            if decTrainMode == "end2end":
                decOptimizer.zero_grad()
                                        
            # Forward pass
            latentL = interpEncoder(sparseL)
            predL = aeDecoder(latentL)
            
            # Get loss
            loss = torch.mean(get_lsd(predL, targetL))
            
            # Calculate new grad
            loss.backward()
            
            # update weights
            if decTrainMode == "end2end":
                decOptimizer.step()
            interpOptimizer.step()
            
            lossEp+=loss.item()
        
        lossEp = lossEp/(batchNo+1)
        
        return lossEp
    
    def get_val_loss():
        lossEp = 0
        for batchNo, batch in enumerate(valDL):
            # Get batch
            if hrtfType is "hrtf":
                targetL = batch["hrtf_l"].to(device)
                sparseL = batch["sparse_hrtf_l"].to(device)
            elif hrtfType == "hrir":
                targetL = batch["hrir_l"].to(device)
                sparseL = batch["sparse_hrtf_l"].to(device)
                                        
            # Forward pass
            latentL = interpEncoder(sparseL)
            predL = aeDecoder(latentL)
            
            # Get loss
            loss = torch.mean(get_lsd(predL, targetL))
            
            lossEp+=loss.item()
        
        lossEp = lossEp/(batchNo+1)
        
        return lossEp
    
    def decoder_finetune_epoch():
        lossEp = 0
        for batchNo, batch in enumerate(trainDL):
            # Get batch
            if hrtfType is "hrtf":
                targetL = batch["hrtf_l"].to(device)
                sparseL = batch["sparse_hrtf_l"].to(device)
            elif hrtfType == "hrir":
                targetL = batch["hrir_l"].to(device)
                sparseL = batch["sparse_hrtf_l"].to(device)
                
            # Initializations
            # Clear gradients
            interpOptimizer.zero_grad()
            decOptimizer.zero_grad()
                                        
            # Forward pass
            latentL = interpEncoder(sparseL)
            predL = aeDecoder(latentL)
            
            # Get loss
            loss = torch.mean(get_lsd(predL, targetL))
            
            # Calculate new grad
            loss.backward()
            
            # update weights
            decOptimizer.step()
            interpOptimizer.step()
            
            lossEp+=loss.item()
        
        lossEp = lossEp/(batchNo+1)
        
        return lossEp
    
    # Main training loop
    print("Begin training")
    bestLoss = 10000
    bestEp = 0
    for thisEpoch in range(numEpochs):
        lossTrain = encoder_train_epoch()
        interpEncoder.eval()
        if decTrainMode == "end2end":
            aeDecoder.eval()
        lossVal = get_val_loss()
        interpEncoder.train()
        if decTrainMode == "end2end":
            aeDecoder.train()
        
        if hrtfType == "hrtf":
            lossTrain = lossTrain * 6 * 20
            lossVal = lossVal * 6 * 20

        print("Epoch %d :- Train Loss: %.5f, Val Loss: %.5f" % 
              (thisEpoch+1, lossTrain, lossVal)) 

        if lossVal < bestLoss:
            bestEp = thisEpoch + 1
            bestLoss = lossVal

            # To save every successive best model
            if saveFlag:
                if decTrainMode == "end2end":
                    torch.save({'interpEncoder': interpEncoder.state_dict(),
                                'interpOptimizer': interpOptimizer.state_dict(),
                                'interpDecoder': aeDecoder.state_dict(),
                                'decOptimizer': decOptimizer.state_dict()},
                               saveFolder+saveFile%numEpochs)
                else:
                    torch.save({'interpEncoder': interpEncoder.state_dict(),
                                'interpOptimizer': interpOptimizer.state_dict()}, 
                               saveFolder+saveFile%numEpochs)

        writer.add_scalar("Training Loss", lossTrain, thisEpoch)
        writer.add_scalar("Validation Loss", lossVal, thisEpoch)

        if thisEpoch - bestEp > 10000:
            print("Best Epoch: %d, Best Val Loss: %.5f" % (bestEp, bestLoss))
            if saveFlag:
                os.rename(saveFolder+saveFile%numEpochs, saveFolder+saveFile%bestEp)
            break  
    
    if decFTEpochs:
        print("Begin decoder finetuning")
        aeDecoder.train()
        bestLoss = 10000
        bestEp = 0
        for thisFTEpoch in range(decFTEpochs):
            lossTrain = decoder_finetune_epoch()
            interpEncoder.eval()
            aeDecoder.eval()
            lossVal = get_val_loss()
            interpEncoder.train()
            aeDecoder.train()
            if hrtfType == "hrtf":
                lossTrain = lossTrain * 6 * 20
                lossVal = lossVal * 6 * 20
            
            print("Epoch %d :- Train Loss: %.5f, Val Loss: %.5f" % 
                  (thisFTEpoch+1, lossTrain, lossVal)) 

            if lossVal < bestLoss:
                bestEp = thisFTEpoch + 1
                bestLoss = lossVal

                # To save every successive best model
                if saveFlag:
                    torch.save({'interpEncoder': interpEncoder.state_dict(),
                                'decoderFT' : aeDecoder.state_dict(),
                                'interpOptimizer': interpOptimizer.state_dict(),
                                'decFTOptimizer': decOptimizer.state_dict()}, 
                               saveFolder+"ft_"+saveFile%decFTEpochs)

            writer.add_scalar("FT Training Loss", lossTrain, thisFTEpoch)
            writer.add_scalar("FT Validation Loss", lossVal, thisFTEpoch)

            if thisFTEpoch - bestEp > 10000:
                print("Best FT Epoch: %d, Best Val Loss: %.5f" % (bestEp, bestLoss))
                if saveFlag:
                    os.rename(saveFolder+"ft_"+saveFile%decFTEpochs, saveFolder+"ft_"+saveFile%bestEp)
                break  
        
    return interpEncoder, aeDecoder

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

    aeDecoder = AEDecoder(outChannels=gp.inOutChannels, finalActType=gp.decFinalActType, BNFlag=gp.decBNFlag, 
                          modelDepth=gp.modelDepth, featMapsMultFact=gp.featMapsMultFact, numConvBlocks=gp.numConvBlocks,
                          outPadArray=gp.decOutPadArray, dropoutP=gp.dropoutP).to(device)
    
    if gp.interpEncoderOp == "conv":
        interpEncoder = SparseConvEncoder(inChannels=gp.inOutChannels, outChannels=gp.scOutChannels, 
                                          BNFlag=gp.scBNFlag, finalActType=gp.scFinalActType, 
                                          numConvBlocks=gp.scNumConvBlocks, kSizeArray=gp.scKSizeArray, 
                                          strideArray=gp.scStrideArray, padArray=gp.scPadArray, 
                                          dropoutP=gp.interpDropoutP).to(device)
    elif gp.interpEncoderOp == "tconv":
        interpEncoder = SparseTConvEncoder(inChannels=gp.inOutChannels, outChannels=gp.stcOutChannels, 
                                           BNFlag=gp.stcBNFlag, finalActType=gp.stcFinalActType,
                                           numTConvBlocks=gp.stcNumTConvBlocks, kSizeArray=gp.stcKSizeArray, 
                                           strideArray=gp.stcStrideArray, padArray=gp.stcPadArray, 
                                           outPadArray=gp.stcOutPadArray, dropoutP=gp.interpDropoutP).to(device)
    
    if gp.interpDecTrainMode == "ft":
        aeCheckpoint = torch.load(gp.interpAeLoadPath, map_location=device)
        aeDecoder.load_state_dict(aeCheckpoint['decoder'])
        aeDecoder.eval()
    
    
    modelOut = train(interpEncoder=interpEncoder, 
                     aeDecoder=aeDecoder,
                     hrtfType=gp.hrtfType,
                     trainDL=trainDL, 
                     valDL=valDL,
                     encLR=gp.interpEncLR,
                     numEpochs=gp.interpNumEpochs,
                     decLR=gp.interpDecLR,
                     decFTEpochs=gp.interpDecFTEpochs,
                     device=device)