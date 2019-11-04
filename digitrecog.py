import tensorflow as tf
from tkinter import *

import numpy as np

import os
import PIL
from PIL import Image, ImageDraw
import PIL.ImageOps
from keras.models import load_model
import cv2






def main():


    def predict():
        filename = 'file.jpg'
        image1.save(filename)
        img=Image.open('file.jpg')
        img3= PIL.ImageOps.invert(img)
        img3.save('result.jpg')
        img5=cv2.imread('result.jpg',0)

        img5 = cv2.resize(img5,(28, 28)).astype(np.float32)
        img5 = np.expand_dims(img5, axis=0)
        img5 = np.expand_dims(img5, axis=3)

        loaded_model=tf.keras.models.load_model('model.h5')
        pred = loaded_model.predict(img5.reshape(1, 28, 28, 1))

        

        var.set('Prediction :'+str(pred.argmax()))
        
        
        
        os.remove('file.jpg')
        os.remove('result.jpg')
        
    def clr():
        cv.delete('all')
        image1.paste('white', (0,0,200,275))
        var.set('Prediction :')
        
    def paint(event):
        x1, y1 = (event.x), (event.y)
        x2, y2 = (event.x + 1), (event.y + 1)
        cv.create_oval((x1, y1, x2, y2), fill='black',width=10)
        #  --- PIL
        draw.line((x1, y1, x2, y2), fill='black',width=10)
    

    root = Tk()

    cv = Canvas(root, width=200, height=275, bg='white')
    # --- PIL
    image1 = PIL.Image.new('RGB', (200, 275),color='white')
    draw = ImageDraw.Draw(image1)
    # ----
    cv.bind('<B1-Motion>', paint)
    cv.pack()

    btn_save = Button(text="Predict", command=predict)
    btn_save.pack()
    btn_clr = Button(text="Clear", command=clr)
    btn_clr.pack()
    var = StringVar()
    label = Label( root, textvariable=var )
    label.pack()
    root.mainloop()

   









main()
