import os

import pandas as pd
import numpy as np
import cv2
import tensorflow as tf

from tensorflow import keras

from PIL import Image, ImageDraw, ImageOps
from utils import get_data





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

    bg_code = (0, 0, 0)
    char_list, font, sample_character, scale = get_data("english", "standard")
    num_chars = len(char_list)
    num_cols = 250
    image = cv2.imread(pathTraitement +"/" + img_name, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    cell_width = width / num_cols
    cell_height = scale * cell_width
    num_rows = int(height / cell_height)
    if num_cols > width or num_rows > height:
        print("Too many columns or rows. Use default setting")
        cell_width = 6
        cell_height = 12
        num_cols = int(width / cell_width)
        num_rows = int(height / cell_height)
    char_width, char_height = font.getsize(sample_character)
    out_width = char_width * num_cols
    out_height = scale * char_height * num_rows
    out_image = Image.new("RGB", (out_width, out_height), bg_code)
    draw = ImageDraw.Draw(out_image)
    for i in range(num_rows):
        for j in range(num_cols):
            partial_image = image[int(i * cell_height):min(int((i + 1) * cell_height), height),
                            int(j * cell_width):min(int((j + 1) * cell_width), width), :]
            partial_avg_color = np.sum(np.sum(partial_image, axis=0), axis=0) / (cell_height * cell_width)
            partial_avg_color = tuple(partial_avg_color.astype(np.int32).tolist())
            char = char_list[min(int(np.mean(partial_image) * num_chars / 255), num_chars - 1)]
            draw.text((j * char_width, i * char_height), char, fill=partial_avg_color, font=font)

    if "black" == "white":
        cropped_image = ImageOps.invert(out_image).getbbox()
    else:
        cropped_image = out_image.getbbox()
    out_image = out_image.crop(cropped_image)
    out_image.save(pathTraitement + "/" + "bis" + img_name)

    imgBonus = cv2.imread(pathTraitement + "/" + "bis" + img_name, cv2.COLOR_BGR2RGB)

   

    img_counter += 1

    # Display
    cv2.imshow('img', imgBonus)


    imgBonus = cv2.imread('./florence-colgate-england-most-beautiful-face.jpg', 0)
    # if(img_counter%5 == 0): 
    #     cv2.imshow('img' , imgBonus)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        for element in os.listdir("./StockImagesOpenCV/48x48"):
            os.remove("./StockImagesOpenCV/48x48/" + element)
        break
        
# Release the VideoCapture object
cap.release()


