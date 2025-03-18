import numpy as np
import cv2
from shapely.geometry import Point, Polygon, LineString

def read_image(path):
    return cv2.imread(path)

def preprocess_image(image, blur="Gaussian"):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur == "Gaussian":
        blurred_image = cv2.GaussianBlur(grey, (15,15), 4, 4)
    elif blur == "Bilateral":
        blurred_image = cv2.bilateralFilter(grey, 150, 200, 2)
    else:
        print("Invalid value for blur")
        return 0

    return blurred_image

def get_edges(image, threshold=0.3, ksize=5):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize)

    edges = np.sqrt(sobelx**2+sobely**2)
    edges = np.where(edges > edges.max()*threshold, 255, 0)

    return np.uint8(edges)

def get_contours(edges_image, curve_sim=0.01):
    contour_images = []
    contour_shapes = []

    contours, hierarchy = cv2.findContours(edges_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    for contour in contours:
        if len(contour)>300:
            contour_image = np.zeros(edges_image.shape)
            epsilon = curve_sim*cv2.arcLength(contour,True)
            shape = cv2.approxPolyDP(contour,epsilon,True)
            print(shape)
            contour_shapes.append(shape)
            cv2.drawContours(contour_image, [shape], 0, [255,255,255], thickness=cv2.FILLED)
            contour_images.append(contour_image)
            # cv2.drawContours(contour_image_full, [shape], 0, [255,255,255], thickness=cv2.FILLED)

    return contour_images, contour_shapes

def format_polygon_points(contour_shapes):
    polygons = []
    for i in range(len(contour_shapes)):
        shape_points = []
        for j in range(len(contour_shapes[i])):
            shape_points.append((int(contour_shapes[i][j][0][0]), int(contour_shapes[i][j][0][1])))
        #print(shape_points)
        polygons.append(Polygon(shape_points))

    return polygons
    