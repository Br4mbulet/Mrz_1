from instuments import *

def compressImageAndSaveToFile():
    imgPxels, imgHeight, imgWidth = origImageLoad('orig.jpg')
    allPxels = getData(imgPxels)
    rectHeigh, rectWidth, p = inputParametrs()
    L = (imgHeight * imgWidth) // (rectWidth * rectHeigh)
    N = 3 * rectHeigh * rectWidth
    X = splitIntoRect(allPxels, rectHeigh, rectWidth).reshape(L, 1, N)
    w_1 = loadArrFroFile("npy\\first-layer-weights_" + str(rectHeigh) + "_" + str(rectWidth) + "_" + str(p) + ".npy")

    Y = []
    for rectNum in range(L):
        Y.append(X[rectNum] @ w_1)
    print(Y)
    saveArrToFile("npy\\compressed_image_array_" + str(rectHeigh) + "_" + str(rectWidth) + "_" + str(p), Y)
