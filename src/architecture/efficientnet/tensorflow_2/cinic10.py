import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from six.moves import urllib
import tarfile

# Credits to: https://github.com/fazilaltinel/CINIC-10-TFLoader

def loadData(pathToDatasetFolder, oneHot=False):
    """
    pathToDatasetFolder: Parent folder of CINIC-10 dataset folder or CINIC-10.tar.gz file
    oneHot: Label encoding (one hot encoding or not)

    Return: Train, validation and test sets and label numpy arrays
    """
    sourceUrl = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
    pathToFile = downloadDataset(pathToDatasetFolder, "CINIC-10.tar.gz", sourceUrl)

    labelDict = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
                'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
                'truck': 9}

    pathToTrain = os.path.join(pathToFile, "train")
    pathToVal = os.path.join(pathToFile, "valid")
    pathToTest = os.path.join(pathToFile, "test")

    imgNamesTrain = [f for dp, dn, fn in os.walk(os.path.expanduser(pathToTrain)) for f in fn]
    imgDirsTrain = [dp for dp, dn, fn in os.walk(os.path.expanduser(pathToTrain)) for f in fn]
    imgNamesVal = [f for dp, dn, fn in os.walk(os.path.expanduser(pathToVal)) for f in fn]
    imgDirsVal = [dp for dp, dn, fn in os.walk(os.path.expanduser(pathToVal)) for f in fn]
    imgNamesTest = [f for dp, dn, fn in os.walk(os.path.expanduser(pathToTest)) for f in fn]
    imgDirsTest = [dp for dp, dn, fn in os.walk(os.path.expanduser(pathToTest)) for f in fn]

    XTrain = np.empty((len(imgNamesTrain), 32, 32, 3), dtype=np.float32)
    YTrain = np.empty((len(imgNamesTrain)), dtype=np.int32)
    XVal = np.empty((len(imgNamesVal), 32, 32, 3), dtype=np.float32)
    YVal = np.empty((len(imgNamesVal)), dtype=np.int32)
    XTest = np.empty((len(imgNamesTest), 32, 32, 3), dtype=np.float32)
    YTest = np.empty((len(imgNamesTest)), dtype=np.int32)

    print("Loading")

    for i in range(len(imgNamesTrain)):
        # img = plt.imread(os.path.join(imgDirsTrain[i], imgNamesTrain[i]))
        img = misc.imread(os.path.join(imgDirsTrain[i], imgNamesTrain[i]))
        if len(img.shape) == 2:
            XTrain[i, :, :, 2] = XTrain[i, :, :, 1] = XTrain[i, :, :, 0] = img/255.
        else:
            XTrain[i] = img/255.
        YTrain[i] = labelDict[os.path.basename(imgDirsTrain[i])]
    for i in range(len(imgNamesVal)):
        # img = plt.imread(os.path.join(imgDirsVal[i], imgNamesVal[i]))
        img = misc.imread(os.path.join(imgDirsVal[i], imgNamesVal[i]))
        if len(img.shape) == 2:
            XVal[i, :, :, 2] = XVal[i, :, :, 1] = XVal[i, :, :, 0] = img/255.
        else:
            XVal[i] = img/255.
        YVal[i] = labelDict[os.path.basename(imgDirsVal[i])]
    for i in range(len(imgNamesTest)):
        # img = plt.imread(os.path.join(imgDirsTest[i], imgNamesTest[i]))
        img = misc.imread(os.path.join(imgDirsTest[i], imgNamesTest[i]))
        if len(img.shape) == 2:
            XTest[i, :, :, 2] = XTest[i, :, :, 1] = XTest[i, :, :, 0] = img/255.
        else:
            XTest[i] = img/255.
        YTest[i] = labelDict[os.path.basename(imgDirsTest[i])]

    if oneHot:
        YTrain = toOneHot(YTrain, 10)
        YVal = toOneHot(YVal, 10)
        YTest = toOneHot(YTest, 10)

    print("+ Dataset loaded")

    return XTrain, YTrain, XVal, YVal, XTest, YTest


def downloadDataset(dirName, fileName, sourceUrl):
    """
    https://github.com/tflearn/tflearn/blob/master/tflearn/datasets/cifar10.py
    """
    cinicDirName = os.path.join(dirName, "CINIC-10/")
    if not os.path.exists(cinicDirName):
        os.mkdir(cinicDirName)
        pathToFile = os.path.join(dirName, fileName)
        if not os.path.exists(pathToFile):
            print("Downloading")
            pathToFile, _ = urllib.request.urlretrieve(sourceUrl, pathToFile, reporthook)
            print("+ Downloaded")
        untar(pathToFile, cinicDirName)
    else:
        print("+ Dataset already downloaded")
    return cinicDirName


def reporthook(blocknum, blocksize, totalsize):
    """
    reporthook from stackoverflow #13881092
    https://github.com/tflearn/tflearn/blob/master/tflearn/datasets/cifar10.py
    """
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))


def untar(fname, path):
    if (fname.endswith("tar.gz")):
        print("Extracting tar file")
        tar = tarfile.open(fname)
        tar.extractall(path=path)
        tar.close()
        print("+ Extracted")
    else:
        print("Not a tar.gz file")


def toOneHot(y, nb_classes=None):
    """
    https://github.com/tflearn/tflearn/blob/master/tflearn/data_utils.py#L36
    """
    if nb_classes:
        # y = np.asarray(y, dtype='int32')
        if len(y.shape) > 2:
            print("Warning: data array ndim > 2")
        if len(y.shape) > 1:
            y = y.reshape(-1)
        Y = np.zeros((len(y), nb_classes))
        Y[np.arange(len(y)), y] = 1.
        return Y
    else:
        y = np.array(y)
        return (y[:, None] == np.unique(y)).astype(np.float32)
