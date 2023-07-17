from tkinter import *
import numpy as np
import cv2
from PIL import Image


def load():
    img = cv2.imread('./Uncle_Roger.jpg')
    cv2.imshow('Image', img)
    print("Height = ", img.shape[0])
    print("Width = ", img.shape[1])

def separation():
    img = cv2.imread('./Flower.jpg')
    cv2.imshow('Original Image', img)
    b, g, r = cv2.split(img)
    zeros = np.zeros(img.shape[:2], dtype="uint8")
    bb = cv2.merge([b, zeros, zeros])
    rr = cv2.merge([zeros, zeros, r])
    gg = cv2.merge([zeros, g, zeros])
    cv2.imshow('Red Channel', rr)
    cv2.imshow('Blue Channel', bb)
    cv2.imshow('Green Channel', gg)

def flipping():

    img=cv2.imread('./Uncle_Roger.jpg')
    cv2.imshow('Original Image', img)
    cv2.imshow('Flipped Image', cv2.flip(img, 1))

def create(v):
    global imgadd
    img = cv2.imread('./Uncle_Roger.jpg')
    fimg = cv2.flip(img, 1)
    al = cv2.getTrackbarPos('weight', 'Blended Image')
    al = al/255
    imgadd = np.uint8(cv2.addWeighted(img, al, fimg, 1-al, 0))
    cv2.imshow('Blended Image', imgadd)

def blending():
    create(0)
    cv2.namedWindow('Blended Image')
    cv2.createTrackbar('weight', 'Blended Image', 0, 255, create)


def median():
    img = cv2.imread('./Cat.png')
    mimg = cv2.medianBlur(img, 7)
    cv2.imshow('Original Image', img)
    cv2.imshow('Median Image', mimg)

def gaussian():
    img = cv2.imread('./Cat.png')
    cv2.imshow('Original Image', img)
    gimg = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imshow('Gaussian Image', gimg)

def bilateral():
    img = cv2.imread('./Cat.png')
    cv2.imshow('Original Image', img)
    bimg = cv2.bilateralFilter(img, 9, 90, 90)
    cv2.imshow('Bilateral Image', bimg)

def gauss():
    img = cv2.imread('./Chihiro.jpg')
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gimg)
    x, y = np.mgrid[-1:2, -1:2]
    guassian_kernel = np.exp(-(x ** 2 + y ** 2))
    guassian_kernel = guassian_kernel / guassian_kernel.sum()
    img_gau = cv2.filter2D(gimg, -1, guassian_kernel)
    cv2.imshow('Guassian Image', img_gau)

def sobelx():
    img = cv2.imread('./Chihiro.jpg')
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gimg)
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    timg = cv2.filter2D(gimg, -1, kernel)
    cv2.imshow('Sobel X Image', timg)

def sobely():
    img = cv2.imread('./Chihiro.jpg')
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gimg)
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    timg = cv2.filter2D(gimg, -1, kernel)
    cv2.imshow('Sobel Y Image', timg)

def magnitude():
    img = cv2.imread('./Chihiro.jpg')
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gimg)
    mag = np.zeros((gimg.shape[0], gimg.shape[1]))
    xkernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ximg = cv2.filter2D(gimg, -1, xkernel)
    ykernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    yimg = cv2.filter2D(gimg, -1, ykernel)
    for i in range(0, gimg.shape[0]):
        for j in range(0, gimg.shape[1]):
            a = pow(ximg[i, j], 2) + pow(yimg[i, j], 2)
            if a > 0:
                mag[i, j] = int(pow(a, 0.5))
    im = Image.fromarray(mag)
    Image._show(im)

def transformation():
    img = cv2.imread('./Parrot.png')
    cv2.imshow('Original Image', img)
    x = float(tx.get())
    y = float(ty.get())
    scale = float(scaling.get())
    deg = float(degree.get())
    m = np.float32([[1, 0, x], [0, 1, y]])
    rows, cols = img.shape[:2]
    resize = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    shifted = cv2.warpAffine(resize, m, (cols, rows))
    M = cv2.getRotationMatrix2D((160, 84), deg, 1)
    trans = cv2.warpAffine(shifted, M, (cols, rows))
    cv2.imshow('Transformed Image', trans)


window = Tk()
degree = StringVar()
scaling = StringVar()
tx = StringVar()
ty = StringVar()
window.title("HW01")
label1 = Label(window, text='1. Image Processing')
label2 = Label(window, text='2. Image Smoothing')
label3 = Label(window, text='3. Edge Detection')
label4 = Label(window, text='4. Transformation')
btn1 = Button(window, command=load, text="1.1 Load Image")
btn2 = Button(window, command=separation, text="1.2 Color Separation")
btn3 = Button(window, command=flipping, text="1.3 Image Flipping")
btn4 = Button(window, command=blending, text="1.4 Blending")
btn5 = Button(window, command=median, text='2.1 Median Filter')
btn6 = Button(window, command=gaussian, text='2.2 Gaussian Blur')
btn7 = Button(window, command=bilateral, text='2.3 Bilateral Filter')
btn8 = Button(window, command=gauss, text='3.1 Gaussian Blur')
btn9 = Button(window, command=sobelx, text='3.2 Sobel X')
btn10 = Button(window, command=sobely, text='3.3 Sobel Y')
btn11 = Button(window, command=magnitude, text='3.4 Magnitude')
entry1 = Entry(window, textvariable=degree)
entry2 = Entry(window, textvariable=scaling)
entry3 = Entry(window, textvariable=tx)
entry4 = Entry(window, textvariable=ty)
btn12 = Button(window, command=transformation, text='Transformation')
label5 = Label(window, text='Degree')
label6 = Label(window, text='Scaling')
label7 = Label(window, text='Tx')
label8 = Label(window, text='Ty')


label1.grid(row=0, column=0)
btn1.grid(row=1, column=0)
btn2.grid(row=2, column=0)
btn3.grid(row=3, column=0)
btn4.grid(row=4, column=0)
label2.grid(row=0, column=1)
btn5.grid(row=1, column=1)
btn6.grid(row=2, column=1)
btn7.grid(row=3, column=1)
label3.grid(row=0, column=2)
btn8.grid(row=1, column=2)
btn9.grid(row=2, column=2)
btn10.grid(row=3, column=2)
btn11.grid(row=4, column=2)
label4.grid(row=0, column=3)
label5.grid(row=1, column=3)
entry1.grid(row=2, column=3)
label6.grid(row=3, column=3)
entry2.grid(row=4, column=3)
label7.grid(row=5, column=3)
entry3.grid(row=6, column=3)
label8.grid(row=7, column=3)
entry4.grid(row=8, column=3)
btn12.grid(row=9, column=3)

window.mainloop()
