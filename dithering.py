# https://en.wikipedia.org/wiki/Dither

from PIL import Image
from random import random
import numpy as np

source = Image.open("gradient.png", "r")

width, height = source.size

source.show()

pixels = np.array(source).astype(float)
grayscale = np.zeros((height, width), float)

# Grayscale
# https://en.wikipedia.org/wiki/Grayscale
for y in range(height):
    for x in range(width):
        # Clinear
        for color in range(3):
            if (pixels[y][x][color] / 255) <= 0.04045:
                pixels[y][x][color] = (pixels[y][x][color] / 255) / 12.92
            else:
                pixels[y][x][color] = ((pixels[y][x][color]/255 + 0.055) / 1.055) ** 2.4
    
        Ylinear = 0.2126 * pixels[y][x][0] + 0.7152 * pixels[y][x][1] + 0.0722 * pixels[y][x][2]

        if Ylinear <= 0.0031308:
            grayscale[y][x] = 12.92 * Ylinear
        else:
            grayscale[y][x] = 1.055 * Ylinear ** (1/2.4) - 0.055

pixels = grayscale.copy()

# Random dithering
for y in range(height):
    for x in range(width):
        if pixels[y][x] > random():
            pixels[y][x] = 1
        else:
            pixels[y][x] = 0

buffer = Image.new("1", source.size)
buffer.putdata(pixels.flatten())
buffer.show()
buffer.close()
print("Random Dithering")

pixels = grayscale.copy()

# Bayer matrix ordered dithering
# https://blog.42yeah.is/rendering/2023/02/18/dithering.html
# https://en.wikipedia.org/wiki/Ordered_dithering

matrix_n = 4

def create_bayer_matrix(matrix: np.ndarray, n: int):
    if (n == 1):
        return matrix
    
    newmatrix = np.zeros((matrix.shape[0] * 2, matrix.shape[1] * 2), float)
    for y in range(matrix.shape[1]):
        for x in range(matrix.shape[0]):
            cell = matrix[y][x]
            newmatrix[y][x] = 4 * cell
            newmatrix[y][x + matrix.shape[0]] = 4 * cell + 2
            newmatrix[y + matrix.shape[1]][x] = 4 * cell + 3
            newmatrix[y + matrix.shape[1]][x + matrix.shape[0]] = 4 * cell + 1

    return create_bayer_matrix(newmatrix, n - 1)

bayer_matrix = create_bayer_matrix(np.array([[0, 2], [3, 1]], float), matrix_n)
bayer_matrix /= bayer_matrix.shape[0] ** 2

for y in range(height):
    for x in range(width):
        matrix_loc_x = x % bayer_matrix.shape[0]
        matrix_loc_y = y % bayer_matrix.shape[1]
        d = bayer_matrix[matrix_loc_y][matrix_loc_x]
        if pixels[y][x] <= d:
            pixels[y][x] = 0
        else:
            pixels[y][x] = 1

buffer = Image.new("1", source.size)
buffer.putdata(pixels.flatten())
buffer.show()
buffer.close()

print(f"Ordered Dithering  ({bayer_matrix.shape[0]}x{bayer_matrix.shape[1]} Bayer Matrix)")

pixels = grayscale.copy()

# Floyd-Steinberg dithering
# https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering

def find_closest_palette_color(oldpixel):
    return round(oldpixel)

for y in range(height):
    for x in range(width):
        oldpixel = pixels[y][x]
        newpixel = find_closest_palette_color(oldpixel)
        pixels[y][x] = newpixel
        error = oldpixel - newpixel
        
        try:
            pixels[y][x + 1] = pixels[y][x + 1] + error * 7 / 16
            pixels[y + 1][x - 1] = pixels[y + 1][x - 1] + error * 3 / 16
            pixels[y + 1][x] = pixels[y + 1][x] + error * 5 / 16
            pixels[y + 1][x + 1] = pixels[y + 1][x + 1] + error * 1 / 16
        except IndexError:
            pass

buffer = Image.new("1", source.size)
buffer.putdata(pixels.flatten())
buffer.show()
buffer.close()

print("Floyd-Steinberg dithering")

# Halftone

source.close()
