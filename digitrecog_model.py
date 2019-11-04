

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tkinter import *
from tkinter import messagebox

from matplotlib.image import imread
import os
import PIL
from PIL import Image, ImageDraw
import PIL.ImageOps
from keras.models import load_model
import cv2
import pickle

np.random.seed(1337)




def main():
    global arr

    
    def trainmodel():
        global arr
        x_train = arr
        f=open('datainfo.dat','rb')
        Y=pickle.load(f)
        f.close()
        y_train = np.array(Y, dtype='uint8')

        # Reshaping the array to 4-dims so that it can work with the Keras API
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

        input_shape = (28, 28, 1)
        # Making sure that the values are float so that we can get decimal points after division

        x_train = x_train.astype('float32')

        # Normalizing the RGB codes by dividing it to the max RGB value.

        # Importing the required Keras modules containing model and layers
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
        # Creating a Sequential Model and adding the layers
        model = Sequential()
        model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation=tf.nn.softmax))
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x=x_train, y=y_train, epochs=18)

        model.save('model.h5')
    def save():
        if len(entry.get())!=0:
            y='data/'+str(len(os.listdir('data/')))+'.jpg'
            image1.save(y)
            try:
                f=open('datainfo.dat','rb')
                l=pickle.load(f)
                f.close()
                l.append(int(entry.get()))
                os.remove('datainfo.dat')
            except:
                l=[int(entry.get())]
            f=open('datainfo.dat','wb')
            pickle.dump(l,f)
            f.close()
            clr()
            entry.delete(0,len(entry.get()))
        else:
            messagebox.showerror("Error", "Entry box cannot be empty")


    def load():

        i=0
        y=len(os.listdir('data/'))
        global arr
        arr = np.zeros((y, 28, 28, 1), dtype='uint8')
        while i<y:

            filename = 'data/'+str(i)+'.jpg'

            img = Image.open(filename)
            img3 = PIL.ImageOps.invert(img)
            img3.save('result.jpg')
            img5 = cv2.imread('result.jpg', 0)

            img5 = cv2.resize(img5, (28, 28)).astype(np.float32)
            img5 = np.expand_dims(img5, axis=0)
            img5 = np.expand_dims(img5, axis=3)

            arr[i]=img5


            os.remove('result.jpg')
            i+=1

        print('loaded')
    def clr():
        cv.delete('all')
        image1.paste('white', (0, 0, 200, 275))
    

    def paint(event):
        x1, y1 = (event.x), (event.y)
        x2, y2 = (event.x + 1), (event.y + 1)
        cv.create_oval((x1, y1, x2, y2), fill='black', width=10)
        #  --- PIL
        draw.line((x1, y1, x2, y2), fill='black', width=10)
    

    root = Tk()
    la=Label(root,text="Steps.\n 1.Create(optional) \n 2.Load \n 3.Train")
    la.pack()
    
    
    
    
    label3=Label(root,text='###### Create Your Own Dataset ######')
    label3.pack()

    cv = Canvas(root, width=200, height=275, bg='white')
    # --- PIL
    image1 = PIL.Image.new('RGB', (200, 275), color='white')
    draw = ImageDraw.Draw(image1)
    # ----
    cv.bind('<B1-Motion>', paint)
    cv.pack()
    la6=Label(root,text='Enter Digit Drawn')
    la6.pack()
    entry=Entry()
    entry.pack()

    btn_save=Button(text='Save Image',command=save)
    btn_save.pack()
    btn_clr = Button(text="Clear Canvas", command=clr)
    labe=Label(root,text='##################################')

    btn_clr.pack()
    labe.pack()
    btn_load = Button(text="Load Data", command=load)
    btn_load.pack()
    btn_tra = Button(text="Train",command=trainmodel)
    btn_tra.pack()

    root.mainloop()



if __name__=='__main__':
    main()


