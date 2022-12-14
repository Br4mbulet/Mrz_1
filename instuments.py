from PIL import Image as Img
import numpy as np
import matplotlib.pyplot as plt

def origImageLoad(file):
    img = Img.open('orig.jpg', 'r')
    imgPxels = img.load()
    imgHeight, imgWidth = np.size(img, 1), np.size(img, 0)  
    return imgPxels, imgHeight, imgWidth

imgPxels, imgHeight, imgWidth = origImageLoad('orig.jpg')
ALPHA = 0.0003

def splitIntoRect(allPxels, rectHeigh, rectWidth):
    rects = []
    for rowNumPxel in range(imgHeight // rectHeigh):
        for colNumPxel in range(imgWidth // rectWidth):
            rect = fillRectWithPixel(allPxels, rowNumPxel, colNumPxel, rectHeigh, rectWidth)
            rects.append(rect)
    return np.array(rects)


def fillRectWithPixel(allPxels, rowNumPxel, colNumPxel, rectHeigh, rectWidth):
    rect = []
    for rowNumRect in range(rectHeigh):
        for colNumRect in range(rectWidth):
            for color in range(3):
                rect.append(allPxels[rowNumPxel * rectHeigh + rowNumRect, colNumPxel * rectWidth + colNumRect, color])
    return rect


def rectToMatrix(rects, rectHeigh, rectWidth):
    resultMatrix = []
    numRectInRow = imgHeight // rectHeigh
    numRectInCol = imgWidth // rectWidth
    for countRectInRow in range(numRectInRow):
        for numRectRow in range(rectHeigh):
            line = fillInLineForResultMatrix(rects, numRectRow, numRectInCol, countRectInRow, rectWidth)
            resultMatrix.append(line)
    return np.array(resultMatrix)


def getData(pxels):
    allPxels = np.zeros([imgHeight, imgWidth, 3], float)
    for rowNum in range(imgHeight):
        for colNum in range(imgWidth):
            cPxel = list(pxels[colNum, rowNum])
            for i in range(len(cPxel)):
                allPxels[rowNum][colNum][i] = ((2 * cPxel[i]) / 255) - 1
    return allPxels


def fillInLineForResultMatrix(rects, numRectRow, numRectInCol, countRectInRow, rectWidth):
    line = []
    for countRectInCol in range(numRectInCol):
        for numRectCol in range(rectWidth):
            dot = []
            for color in range(3):
                pxelColor = rects[countRectInRow * numRectInCol + countRectInCol, 0, (numRectRow * rectWidth * 3) + (numRectCol * 3) + color]
                dot.append(pxelColor)
            line.append(dot)
    return line



def weightsUpdateSecondLayer(w_2, Y_i, delt_Xi):
    return w_2 - np.matmul((ALPHA * np.transpose(Y_i)), delt_Xi)


def weightsUpdateFirstLayer(w_1, w_2, X_i, delt_Xi):
    trans = np.transpose(X_i)
    return w_1 - np.matmul(np.matmul((ALPHA * trans), delt_Xi), np.transpose(w_2))

    
def weightsNormalization(n_w):
    for colNum in range(len(n_w[0])):
        sum = 0
        for rowNum in range(len(n_w)):
            sum += n_w[rowNum][colNum] ** 2
        sum = sum ** (0.5)
        for rowNum in range(len(n_w)):
            n_w[rowNum][colNum] = n_w[rowNum][colNum] / sum
    return n_w


def saveArrToFile(PASS, arr):
    np.save(PASS, arr)


def loadArrFroFile(PASS):
    return np.load(PASS)


def saveImgAndShow(arrImg, nameSave):
    readImg = 1 * (1 + arrImg) / 2
    plt.axis('off')
    plt.imshow(readImg)
    plt.savefig(nameSave, transparent=True)
    plt.show()


def training(X, L, N, p, rectHeigh, rectWidth, shouldNormal, epochNum):
    w_1 = np.random.uniform(-1, 1, size=(N, p))
    w_2 = w_1.transpose()
    errors = np.zeros(L)

    iter = 0
    for _ in range(epochNum):
        errSum = 0
        iter = iter + 1
        for rectNum in range(L):
            Y_i = X[rectNum] @ w_1
            X_e = Y_i @ w_2
            delt_Xi = X_e - X[rectNum]

            w_1 = weightsUpdateFirstLayer(
                w_1, w_2, X[rectNum], delt_Xi)
            w_2 = weightsUpdateSecondLayer(w_2, Y_i, delt_Xi)

            if shouldNormal:
                w_1 = weightsNormalization(w_1)
                w_2 = weightsNormalization(w_2)

            errors[rectNum] = (delt_Xi ** 2).sum()
        errSum = np.sum(errors)
        print(f"Iteration №{iter}   Error sum: {errSum}")
    saveArrToFile("npy\\first-layer-weights_" + str(rectHeigh) + "_" + str(rectWidth) + "_" + str(p), w_1)
    saveArrToFile("npy\\second-layer-weights_" + str(rectHeigh) + "_" + str(rectWidth) + "_" + str(p), w_2)
    return w_1, w_2


def inputParametrs():
    print('Rows number in rectangle (m)')
    rectHeigh = int(input())
    print('Columns number in rectangle (n)')
    rectWidth = int(input())
    print('Neurals number on the layer (p)')
    p = int(input())
    return rectHeigh, rectWidth, p