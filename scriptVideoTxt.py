import pandas as pd
import numpy as np
import cv2
import argparse

import os



cam = cv2.VideoCapture(0)


cv2.namedWindow("test")

img_counter = 0

pathTraitement = "./StockImagesOpenCV/traitementImage"

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()

    _, imgCam = cap.read()


    # Draw the rectangle around each face
    img_name = "opencv_frame_{}.jpg".format(img_counter)
    
    cv2.imwrite(os.path.join(pathTraitement , img_name), img) 

    if "simple" == "simple":
        CHAR_LIST = '@%#*+=-:. '
    else:
        CHAR_LIST = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
    num_chars = len(CHAR_LIST)
    num_cols = 300
    image = cv2.imread(pathTraitement +"/" + img_name, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape
    cell_width = width / num_cols
    cell_height = 2 * cell_width
    num_rows = int(height / cell_height)

    
    output_file = open("output.txt", 'w')
    for i in range(num_rows):
        for j in range(num_cols):
            output_file.write(
                CHAR_LIST[min(int(np.mean(image[int(i * cell_height):min(int((i + 1) * cell_height), height),
                                          int(j * cell_width):min(int((j + 1) * cell_width),
                                                                  width)]) * num_chars / 255), num_chars - 1)])
        output_file.write("\n")
    output_file.close()

    with open('output.txt', 'r') as f:
        print(f.read())

    # result = "" 

    # for i in range(num_rows):
    #     for j in range(num_cols):
    #         result += CHAR_LIST[min(int(np.mean(image[int(i * cell_height):min(int((i + 1) * cell_height), height),
    #                                       int(j * cell_width):min(int((j + 1) * cell_width),
    #                                                               width)]) * num_chars / 255), num_chars - 1)]
        
    # print(result)        
    # print("\n")
  
       
    

   

    img_counter += 1

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
# Release the VideoCapture object
cap.release()


