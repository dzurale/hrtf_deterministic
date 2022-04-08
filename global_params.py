# Global global
dataPath = "../data/cipic_latest.hdf5"
logFolder = "./logs/"
exp = "exp5"
hrtfType = "hrtf"
whichHRIR = "raw"
valSubjs = [12, 29, 40]
anthroOp = None
if whichHRIR == "trunc64":
    anthroFlag = True
    if hrtfType == "hrtf":
        inOutChannels = 32
    elif hrtfType == "hrir":
        inOutChannels = 64
elif whichHRIR == "raw":
    if anthroOp is not None:
        anthroFlag = True
    else:
        anthroFlag = False
    if hrtfType == "hrtf":
        inOutChannels = 128
    elif hrtfType == "hrir":
        inOutChannels = 200

saveFlag = True
saveFolder = "./saved_models/"
saveFile = "ae_"+exp+"_ep%d.pth"

# Dataset parameters
azOps = [6, 1]
elOps = [12, 2]
samplingOp = "uni"
if elOps[1] > 7:
    quadSelect = "upper"
else:
    quadSelect = "full"
        

# Model specs        
# global
rootFeatMaps = 128
convGroups = 8

# passable
modelDepth = 3
featMapsMultFact = 1
numConvBlocks = 0

# encoder specific
encDSType = "conv"
encBNFlag = False

# decoder specific
decFinalActType = "sigmoid"
decBNFlag = False
decOutPadArray = [(1, 0), (0, 1), 0]
#decOutPadArray = [(1, 0), (1, 0), 0]

# Train parameters
encLR = 0.0001
decLR = encLR
numEpochs = 400000
dropoutP = 0

# Parameters to resume training
resumeLoadPath = saveFolder+"ae_"+exp+"_ep%d.pth" % 400000
resumeSaveFile = "resumed_ae_"+exp+"_ep%d.pth"

##################################################################################
##################################################################################
# Interpolation specific
# Global
interpVer = "-4"
interpSaveFile = "interp_ae_"+exp+interpVer+"_ep%d.pth"
interpAeLoadPath = saveFolder+"resumed_ae_"+exp+"_ep123394.pth"

# Sparse Conv Encoder Model parameters
scOutChannels = 128
scFinalActType = "elu"
scBNFlag = False
scNumConvBlocks = 1
scKSizeArray = [3, 3]
scStrideArray = [1, 1]
scPadArray = [1, 1]

# Train parameters
interpEncLR = 0.0001
interpDecFTLR = interpEncLR/100
interpNumEpochs = 400000
interpDecFTEpochs = 0
interpDropoutP = 0
interpDecTrainMode = "full"
