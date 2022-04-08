import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import global_params as gp

rootFeatMaps = gp.rootFeatMaps
convGroups = gp.convGroups

class AEEncoder(nn.Module):
    def __init__(self, inChannels, dsType="conv", BNFlag=False,
                 modelDepth=3, featMapsMultFact=1, numConvBlocks=0, dropoutP=0):
        super(AEEncoder, self).__init__()
        
        self.dsType = dsType
        self.BNFlag = BNFlag
        self.modelDepth = modelDepth
        self.numConvBlocks = numConvBlocks
        self.dropoutP = dropoutP
        self.moduleDict = nn.ModuleDict()
        
        for depth in range(modelDepth - 1):
            outFeatMaps = int((featMapsMultFact ** (depth+1)) * rootFeatMaps)
            
            if depth == 0:
                self.firstConv = nn.Conv2d(in_channels=inChannels, out_channels=rootFeatMaps, 
                                           kernel_size=3, stride=1, padding=1,
                                           groups=convGroups)
                self.moduleDict["conv_first"] = self.firstConv
                
                if BNFlag:
                    self.batchNorm = nn.BatchNorm2d(num_features=rootFeatMaps)
                    self.moduleDict["bn_first"] = self.batchNorm
                    
                inChannels=rootFeatMaps
                    
            for convNo in range(numConvBlocks):
                self.convBlock = nn.Conv2d(in_channels=inChannels, out_channels=outFeatMaps, 
                                           kernel_size=3, stride=1, padding=1, 
                                           groups=convGroups) 
                self.moduleDict["conv_{}_{}".format(depth, convNo)] = self.convBlock
                
                if BNFlag:
                    self.batchNorm = nn.BatchNorm2d(num_features=outFeatMaps)
                    self.moduleDict["bn_{}_{}".format(depth, convNo)] = self.batchNorm
                
                inChannels = outFeatMaps
            
            if dsType == "pooling":
                self.downsampling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                self.moduleDict["ds_{}".format(depth)] = self.downsampling

            elif dsType == "conv":
                self.downsampling = nn.Conv2d(in_channels=inChannels, out_channels=outFeatMaps, 
                                              kernel_size=2, stride=2, padding=0, 
                                              groups=convGroups)
                self.moduleDict["ds_{}".format(depth)] = self.downsampling

                if BNFlag:
                    self.batchNorm = nn.BatchNorm2d(num_features=outFeatMaps)
                    self.moduleDict["dsbn_{}".format(depth)] = self.batchNorm
                        
            inChannels=outFeatMaps
            
        self.finalConv = nn.Conv2d(in_channels=outFeatMaps, out_channels=outFeatMaps, 
                                kernel_size=3, stride=1, padding=1,
                                groups=convGroups)
                        
        if dropoutP:        
            self.dropout = nn.Dropout(p=dropoutP)
            
                
    def forward(self, x):
        for k, op in self.moduleDict.items():
            if k.startswith("conv"):
                if self.dropoutP:
                    x = self.dropout(x)
                x = op(x)
                if not self.BNFlag:
                    x = F.elu(x)                
            elif k.startswith("bn"):                
                x = op(x)
                x = F.elu(x)
            elif k.startswith("ds"):
                if self.dsType == "conv":
                    if self.dropoutP:
                        x = self.dropout(x)
                    x = op(x)
                    x = F.elu(x)
                else:    
                    x = op(x)
            elif k.startswith("dsbn"):
                x = op(x)
                x = F.elu(x)

        x = F.elu(self.finalConv(x))
        
        return x
    
class AEDecoder(nn.Module):
    def __init__(self, outChannels, finalActType="sigmoid", BNFlag=True, 
                 modelDepth=3, featMapsMultFact=1, numConvBlocks=0, outPadArray=[], dropoutP=0):
        super(AEDecoder, self).__init__()
        
        self.BNFlag = BNFlag
        self.numConvBlocks = numConvBlocks
        self.dropoutP = dropoutP
        self.moduleDict = nn.ModuleDict()
        
        for depth in range(modelDepth - 2, -1, -1):
            inChannels = int((featMapsMultFact ** (depth + 1)) * rootFeatMaps)
            if numConvBlocks:
                outFeatMaps = inChannels
            else:
                outFeatMaps = int((featMapsMultFact ** depth) * rootFeatMaps)
                
            if depth == modelDepth - 2:
                self.firstConv = nn.Conv2d(in_channels=inChannels, out_channels=inChannels,
                                           kernel_size=3, stride=1, padding=1,
                                           groups=convGroups)
                self.moduleDict["conv_first"] = self.firstConv
                
                if BNFlag:
                    self.batchNorm = nn.BatchNorm2d(num_features=inChannels)
                    self.moduleDict["bn_first"] = self.batchNorm
                
            self.tConvBlock = nn.ConvTranspose2d(in_channels=inChannels, out_channels=outFeatMaps,
                                                 kernel_size=2, stride=2, padding=0,
                                                 output_padding=outPadArray[depth], 
                                                 groups=convGroups)
            self.moduleDict["tconv_{}".format(depth+1)] = self.tConvBlock
            
            if BNFlag:
                self.batchNorm = nn.BatchNorm2d(num_features=outFeatMaps)
                self.moduleDict["tbn_{}".format(depth+1)] = self.batchNorm
            
            outFeatMaps = int((featMapsMultFact ** depth) * rootFeatMaps)
            for convNo in range(numConvBlocks):    
                self.convBlock = nn.Conv2d(in_channels=inChannels, out_channels=outFeatMaps, 
                                           kernel_size=3, stride=1, padding=1,
                                           groups=convGroups)
                self.moduleDict["conv_{}_{}".format(depth, convNo)] = self.convBlock
                
                if BNFlag:
                    self.batchNorm = nn.BatchNorm2d(num_features=outFeatMaps)
                    self.moduleDict["bn_{}_{}".format(depth, convNo)] = self.batchNorm
                    
                inChannels = outFeatMaps
                
        self.finalConv = nn.Conv2d(in_channels=outFeatMaps, out_channels=outChannels,
                                   kernel_size=3, stride=1, padding=1,
                                   groups=convGroups)
        self.moduleDict["final_conv"] = self.finalConv
        
#         self.finalTConv = nn.ConvTranspose2d(in_channels=outFeatMaps, out_channels=outChannels,
#                                              kernel_size=3, stride=1, padding=0,
#                                              groups=convGroups)
#         self.moduleDict["final_tconv"] = self.finalTConv
        
        if dropoutP:
            self.dropout = nn.Dropout(p=dropoutP)
        
        if finalActType == "sigmoid":
            self.finalAct = nn.Sigmoid()
        else:
            self.finalAct = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        for k, op in self.moduleDict.items():
            if k.startswith("tconv"):
                if self.dropoutP:
                    x = self.dropout(x)
                x = op(x)
                if not self.BNFlag:
                    x = F.elu(x)
            elif k.startswith("tbn"):
                x = op(x)
                x = F.elu(x)
            elif k.startswith("conv"):
                if self.dropoutP:
                    x = self.dropout(x)
                x = op(x)
                if not self.BNFlag:
                    x = F.elu(x)
            elif k.startswith("bn"):
                x = op(x)
                x = F.elu(x)
            elif k.startswith("final"):
                if self.dropoutP:
                    x = self.dropout(x)
                x = op(x)
                x = self.finalAct(x)
                
        return x
    
class SparseConvEncoder(nn.Module):
    def __init__(self, inChannels, outChannels, 
                 BNFlag=False, finalActType=None,
                 numConvBlocks=1, kSizeArray=[3], strideArray=[1], padArray=[1], 
                 dropoutP=0):
        super(SparseConvEncoder, self).__init__()
        self.dropoutP = dropoutP
        self.finalActType = finalActType
        self.numConvBlocks = numConvBlocks
        self.moduleDict = nn.ModuleDict()
        
        outFeatMaps = rootFeatMaps
        
        if numConvBlocks > 1:
            for convIdx in range(numConvBlocks):            
                if convIdx == 0:
                    self.conv = nn.Conv2d(in_channels=inChannels, 
                                          out_channels=outFeatMaps, 
                                          kernel_size=kSizeArray[convIdx], 
                                          stride=strideArray[convIdx],
                                          padding=padArray[convIdx],
                                          groups=convGroups)
                    self.moduleDict["conv_{}".format(convIdx)] = self.conv

                    if BNFlag:
                        self.batchNorm = nn.BatchNorm2d(num_features=outFeatMaps)
                        self.moduleDict["bn_{}".format(convIdx)] = self.batchNorm

                elif convIdx == numConvBlocks - 1:
                    self.conv = nn.Conv2d(in_channels=outFeatMaps, 
                                          out_channels=outChannels, 
                                          kernel_size=kSizeArray[convIdx], 
                                          stride=strideArray[convIdx],
                                          padding=padArray[convIdx],
                                          groups=convGroups)
                    self.moduleDict["conv_{}".format(convIdx)] = self.conv

                else:
                    self.conv = nn.Conv2d(in_channels=outFeatMaps, 
                                          out_channels=outFeatMaps, 
                                          kernel_size=kSizeArray[convIdx], 
                                          stride=strideArray[convIdx],
                                          padding=padArray[convIdx],
                                          groups=convGroups)
                    self.moduleDict["conv_{}".format(convIdx)] = self.conv

                    if BNFlag:
                        self.batchNorm = nn.BatchNorm2d(num_features=outFeatMaps)
                        self.moduleDict["bn_{}".format(convIdx)] = self.batchNorm
        else:
            self.conv = nn.Conv2d(in_channels=inChannels, 
                                  out_channels=outChannels, 
                                  kernel_size=kSizeArray[0], 
                                  stride=strideArray[0],
                                  padding=padArray[0],
                                  groups=convGroups)
            self.moduleDict["conv_{}".format(0)] = self.conv
                
        if dropoutP:
            self.dropout = nn.Dropout(p=dropoutP)
        
        if finalActType == "sigmoid":
            self.finalAct = nn.Sigmoid()
                
    
    def forward(self, x):
        for k, op in self.moduleDict.items():
            if k.startswith("conv"):
                if self.dropoutP:
                    self.dropout(x)
                x = op(x)
                if k.endswith(str(self.numConvBlocks-1)):
                    if self.finalActType is not None:
                        if self.finalActType == "sigmoid":
                            x = self.finalAct(x)
                        else:
                            x = F.elu(x)
                else:
                    if not self.BNFlag:
                        x = F.elu(x)
                        
            elif k.startswith("bn"):
                x = op(x)
                x = F.elu(x)
                
        return x
    
class SparseTConvEncoder(nn.Module):
    def __init__(self, inChannels, outChannels, 
                 BNFlag=False, finalActType=None,
                 numTConvBlocks=1, kSizeArray=[2], strideArray=[2], padArray=[0], outPadArray=[0],
                 dropoutP=0):
        super(SparseTConvEncoder, self).__init__()
        self.dropoutP = dropoutP
        self.finalActType = finalActType
        self.numTConvBlocks = numTConvBlocks
        self.moduleDict = nn.ModuleDict()
        
        outFeatMaps = rootFeatMaps
        
        if numTConvBlocks > 1:
            for tconvIdx in range(numTConvBlocks):            
                if tconvIdx == 0:
                    self.tconv = nn.ConvTranspose2d(in_channels=inChannels, 
                                                    out_channels=outFeatMaps, 
                                                    kernel_size=kSizeArray[tconvIdx], 
                                                    stride=strideArray[tconvIdx],
                                                    padding=padArray[tconvIdx],
                                                    output_padding=outPadArray[tconvIdx],
                                                    groups=convGroups)
                    self.moduleDict["tconv_{}".format(tconvIdx)] = self.tconv

                    if BNFlag:
                        self.batchNorm = nn.BatchNorm2d(num_features=outFeatMaps)
                        self.moduleDict["bn_{}".format(tconvIdx)] = self.batchNorm

                elif tconvIdx == numTConvBlocks - 1:
                    self.tconv = nn.ConvTranspose2d(in_channels=outFeatMaps, 
                                                    out_channels=outChannels, 
                                                    kernel_size=kSizeArray[tconvIdx], 
                                                    stride=strideArray[tconvIdx],
                                                    padding=padArray[tconvIdx],
                                                    output_padding=outPadArray[tconvIdx],
                                                    groups=convGroups)
                    self.moduleDict["tconv_{}".format(tconvIdx)] = self.tconv

                else:
                    self.tconv = nn.ConvTranspose2d(in_channels=outFeatMaps, 
                                                    out_channels=outFeatMaps, 
                                                    kernel_size=kSizeArray[tconvIdx], 
                                                    stride=strideArray[tconvIdx],
                                                    padding=padArray[tconvIdx],
                                                    output_padding=outPadArray[tconvIdx],
                                                    groups=convGroups)
                    self.moduleDict["tconv_{}".format(tconvIdx)] = self.tconv

                    if BNFlag:
                        self.batchNorm = nn.BatchNorm2d(num_features=outFeatMaps)
                        self.moduleDict["bn_{}".format(tconvIdx)] = self.batchNorm
        else:
            self.tconv = nn.ConvTranspose2d(in_channels=inChannels, 
                                            out_channels=outChannels, 
                                            kernel_size=kSizeArray[tconvIdx], 
                                            stride=strideArray[tconvIdx],
                                            padding=padArray[tconvIdx],
                                            output_padding=outPadArray[tconvIdx],
                                            groups=convGroups)
            self.moduleDict["tconv_{}".format(0)] = self.tconv
                
        if dropoutP:
            self.dropout = nn.Dropout(p=dropoutP)
        
        if finalActType == "sigmoid":
            self.finalAct = nn.Sigmoid()
                
    
    def forward(self, x):
        for k, op in self.moduleDict.items():
            if k.startswith("tconv"):
                if self.dropoutP:
                    self.dropout(x)
                x = op(x)
                if k.endswith(str(self.numTConvBlocks-1)):
                    if self.finalActType is not None:
                        if self.finalActType == "sigmoid":
                            x = self.finalAct(x)
                        else:
                            x = F.elu(x)
                else:
                    if not self.BNFlag:
                        x = F.elu(x)
                        
            elif k.startswith("bn"):
                x = op(x)
                x = F.elu(x)
                
        return x
    
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
