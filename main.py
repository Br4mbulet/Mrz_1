from instuments import origImageLoad
from training import trainAndSaveResultImage
from decompression import decompressImageAndSaveImage
from compression import compressImageAndSaveToFile

if __name__ == '__main__':

    fileName = 'orig.jpg'
    imgPxels, imgHeight, imgWidth = origImageLoad(fileName)
    ALPHA = 0.0003

    while True:
        action = int(input("Select action: \n"
                           "1. Training \n"
                           "2. Compressing image \n"
                           "3. Decompressing image\n"
                           "4. Exit\n"))
        if action == 1:
            trainAndSaveResultImage()
        elif action == 2:
            compressImageAndSaveToFile()
        elif action == 3:
            decompressImageAndSaveImage()
        elif action == 4:
            exit()
        else:
            print('Wrong action')
