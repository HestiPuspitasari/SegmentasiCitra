import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("Img/Lemon.jpg")

color = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.figure('Grayscale')
plt.imshow(gray, cmap='gray')

ret, thresh = cv.threshold(gray, 136, 255, cv.THRESH_BINARY_INV)
plt.figure('Binary')
plt.imshow(thresh, cmap='gray')

kernel = np.ones((8, 8), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 7)
plt.figure('Noises Cleaned')
plt.imshow(opening, cmap='gray')

fg = cv.erode(opening, kernel, iterations = 7)
ret, fg = cv.threshold(fg, 0.6 * fg.max(), 255, 0)
fg = np.uint8(fg)
plt.figure('Foreground')
plt.imshow(fg, cmap='gray')

ret, markers = cv.connectedComponents(fg)
markers = markers + 10
markers = cv.watershed(color, markers)
color[markers == 1] = [0, 255, 0]
plt.figure('Markers')
plt.imshow(markers, cmap='gray')
plt.figure('Asli')
plt.imshow(color)
plt.show()
