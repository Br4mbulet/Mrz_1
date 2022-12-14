from instuments import *

def decompressImageAndSaveImage():
    imgPxels, imgHeight, imgWidth = origImageLoad('orig.jpg')
    rectHeigh, rectWidth, p = inputParametrs()
    L = (imgHeight * imgWidth) // (rectWidth * rectHeigh)
    Y = loadArrFroFile("npy\\compressed_image_array_" + str(rectHeigh) + "_" + str(rectWidth) + "_" + str(p) + ".npy")
    w_2 = loadArrFroFile("npy\\second-layer-weights_" + str(rectHeigh) + "_" + str(rectWidth) + "_" + str(p) + ".npy")
    result = []
    for rectNum in range(L):
        result.append(Y[rectNum] @ w_2)
    result = np.array(result)
    saveImgAndShow(rectToMatrix(result, rectHeigh, rectWidth),"img\\image_output_decompressed_" + str(rectHeigh) + "_" + str(rectWidth) + "_" + str(p) + ".png")
