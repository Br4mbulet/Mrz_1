from instuments import *

def trainAndSaveResultImage():
    imgPxels, imgHeight, imgWidth = origImageLoad('orig.jpg')
    rectHeigh, rectWidth, p = inputParametrs()
    L = (imgHeight * imgWidth) // (rectWidth * rectHeigh)
    N = 3 * rectHeigh * rectWidth

    print('Epoch number')
    epochNum = int(input())

    #print('Train mode: 1 - with normalization, 0 - without normalization')
    shouldNormal = 0 #bool(int(input()))

    allPxels = getData(imgPxels)
    rects = splitIntoRect(allPxels, rectHeigh, rectWidth).reshape(L, 1, N)
    saveImgAndShow(allPxels, "img\\image_output.png")
    w_1, w_2 = training(rects, L, N, p, rectHeigh, rectWidth, shouldNormal, epochNum)

    result = []
    for rects in rects:
        result.append(np.matmul(np.matmul(rects, w_1), w_2))
    result = np.array(result)

    saveImgAndShow(rectToMatrix(result, rectHeigh, rectWidth), "img\\image_output1.png")
