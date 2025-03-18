import cv2
import numpy as np
import math
import GlobalNavigation as globnav
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
import constants
import matplotlib.pyplot as plt

"""
To release the camera at the end of the execution of everything
"""
def release_camera(cam):
    cam.release()
    cv2.destroyAllWindows() 
    return 0

def get_bounds(frame, detector):
    marker_corners, marker_ids, _ = detector.detectMarkers(frame)
    for corners, id in zip(marker_corners, marker_ids):
        if id == constants.TOP_LEFT_MARKER:
            (tl, tr, br, bl) = corners[0]
            top_left = (int((tl[0] + br[0])/2), int((tl[1] + br[1])/2))
        elif id == constants.TOP_RIGHT_MARKER:
            (tl, tr, br, bl) = corners[0]
            top_right = (int((tl[0] + br[0])/2), int((tl[1] + br[1])/2))
        elif id == constants.BOTTOM_LEFT_MARKER:
            (tl, tr, br, bl) = corners[0]
            bottom_left = (int((tl[0] + br[0])/2), int((tl[1] + br[1])/2))
        elif id == constants.BOTTOM_RIGHT_MARKER:
            (tl, tr, br, bl) = corners[0]
            bottom_right = (int((tl[0] + br[0])/2), int((tl[1] + br[1])/2))

    x_lower = max(top_left[0], bottom_left[0])
    x_upper = min(top_right[0], bottom_right[0])
    y_lower = max(top_left[1], top_right[1])
    y_upper = min(bottom_left[1], bottom_right[1])

    return x_lower, x_upper, y_upper, y_lower
"""
Preprocesses an image that comes in through the camera (or any image for that matter)

It applies:
    - A red threshold to see only the global obstacles (hue from 0-10 and from 160-180)
    - Removes a white threshold to remove any influence of the goal marker
    - Blurs the images to fliter out any noise
        - Large filter for filtering out things like the lines between the blue planks should they be there.
    
return: image with only the shapes corresponding to the global obstacles
    
"""
def preprocess_image(image, blur="Gaussian"):
    frame_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0,50,50])
    upper_red1 = np.array([10,255,255])

    lower_red2 = np.array([160,50,50])
    upper_red2 = np.array([180,255,255])

    lower_v = np.array([0, 0, 0])
    upper_v = np.array([255, 255, constants.V_TRESH])

    mask1 = cv2.inRange(frame_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(frame_hsv, lower_red2, upper_red2)
    mask_v = cv2.inRange(frame_hsv, lower_v, upper_v)

    cv2.imwrite("maskv.jpeg", mask_v)
    cv2.imwrite("mask_red.jpeg", mask1 + mask2)

    mask = mask1 + mask2 - mask_v
    mask = np.clip(mask, 0, None)

    mask_blurred = cv2.GaussianBlur(mask, (15,15), 2, 2)

    shapes = np.where(mask_blurred >= 255, mask_blurred, 0)
    return shapes

"""
Gets the edges of image - used to find the global obstacle edges from the preprocessed image
"""
def get_edges(image, threshold=0.3, ksize=5):
    edges = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,2)
    edges = np.uint8(edges)
    return edges

"""
Finds the contours of the edges map to convert the shapes into polygons that we can work with.
"""
def get_contours(edges_image, curve_sim=0.01):
    contour_shapes = []

    contours, hierarchy = cv2.findContours(edges_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        if len(contour)>300:
            epsilon = curve_sim*cv2.arcLength(contour,True)
            shape = cv2.approxPolyDP(contour,epsilon,True)
            contour_shapes.append(shape)

    return contour_shapes

"""
Finds global obstacles from an image frame
"""
def find_obstacles(frame):
    shapes = preprocess_image(frame)
    edges = get_edges(shapes)

    contour_shapes = get_contours(edges)
    return contour_shapes

"""
Draws the obstacles for a given set of obstacle contours
"""
def draw_obstacles(frame, shapes) :
    for shape in shapes:
        cv2.drawContours(frame, [shape], 0, [255,0,0], thickness=cv2.FILLED)

"""
Formats the contour points into a suitable form for the path finding algorithm
"""
def format_polygon_points(contour_shapes):
    polygons = []
    for i in range(len(contour_shapes)):
        shape_points = []
        for j in range(len(contour_shapes[i])):
            shape_points.append((int(contour_shapes[i][j][0][0]), int(contour_shapes[i][j][0][1])))
        polygons.append(Polygon(shape_points))

    return polygons

"""
Draws the detected markers on a blank frame and on a frame
"""
def draw_codes(frame, blank_frame, markerCorners, markerIds):
    position = tuple([0,0])
    direction_vector = tuple([0,0])

    if len(markerCorners) > 0:
            ids = markerIds.flatten()

            for (corners, id) in zip(markerCorners, ids):
                corners = corners.reshape((4,2))
                (top_left, top_right, bottom_right, bottom_left) = corners

                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))

                centre_x = 0.5*(top_left[0] + bottom_right[0])
                centre_y = 0.5*(top_left[1] + bottom_right[1])

                centre = tuple([int(centre_x), int(centre_y)])

                # Draw box around the code in MAIN CAMERA frame
                cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(frame, bottom_left, top_left, (255, 0, 0), 2)

                # Draw box around the code in MAP frame
                cv2.line(blank_frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(blank_frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(blank_frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(blank_frame, bottom_left, top_left, (255, 0, 0), 2)

                # Mark the corners of the shapes in the main frame
                cv2.putText(frame, 'TL', top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 4)
                cv2.putText(frame, 'TR', top_right, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 4)
                cv2.putText(frame, 'BR', bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 4)
                cv2.putText(frame, 'BL', bottom_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 4)

                dir_x = int(top_left[0] - bottom_left[0])
                dir_y = int(top_left[1] - bottom_left[1])

                direction = [dir_x, dir_y]
                unit_direction = direction / np.linalg.norm(direction)

                cv2.putText(frame, "direction: {}".format(unit_direction), centre, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.line(frame, centre, tuple([int(top_left[0] + top_right[0])//2, int(top_left[1] + top_right[1])//2]), [0,0,255], 4)
                # cv2.putText(blank_frame, "{}".format(id), centre, cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

                if id == constants.THYMIO_ID:
                    position = centre
                    direction_vector = direction
                    cv2.putText(blank_frame, "THYMIO", centre, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

    return position, direction_vector

"""
Gets the centre and direction of the thymio given its marker details
"""
def get_thymio_params(marker_corners, marker_ids):
    centre = tuple([-1,-1])
    unit_direction = tuple([0,0])
    markers = marker_ids.flatten()
    for (corners, id) in zip(marker_corners, markers):

        if id == constants.THYMIO_ID:
            (top_left, top_right, bottom_right, bottom_left) = corners

            centre_x = 0.5*(top_left[0] + bottom_right[0])
            centre_y = 0.5*(top_left[1] + bottom_right[1])
            centre = tuple([int(centre_x), int(centre_y)])

            dir_x = int(top_left[0] - bottom_left[0])
            dir_y = int(top_left[1] - bottom_left[1])
            direction = [dir_x, dir_y]
            unit_direction = direction / np.linalg.norm(direction)

    return centre, unit_direction

"""
Returns the variance for a list of angles
"""            
def get_variance_angle(angles):
    return np.var(angles)

"""
Finds the position of the goal and the thymio given their aruco marker ids 
"""
def live_pos(frame, detector, thymio_id, goal_id):
    g_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    marker_corners, marker_ids, _ = detector.detectMarkers(g_frame)

    thymio_centre = None
    angle = None
    thymio_corners = None
    goal = None
    dir_vec = None

    if len(marker_corners) > 0:
        for (corners, id) in zip(marker_corners, marker_ids):
            if id == thymio_id:
                thymio_corners = corners[0]
                (top_left, top_right, bottom_right, bottom_left) = corners[0]

                centre_x = 0.5*(top_left[0] + bottom_right[0])
                centre_y = 0.5*(top_left[1] + bottom_right[1])
                thymio_centre = [int(centre_x), int(centre_y)]

                direction_x = top_left[0] - bottom_left[0]
                direction_y = top_left[1] - bottom_left[1]
                dir_vec = [direction_x, direction_y]
                if np.linalg.norm(dir_vec) != 0:
                    dir_vec = dir_vec / np.linalg.norm(dir_vec)
                angle = math.atan2(direction_y, direction_x)

            if id == goal_id:
                goal_corners = corners[0]
                (top_left, top_right, bottom_right, bottom_left) = corners[0]

                centre_g_x = 0.5*(top_left[0] + bottom_right[0])
                centre_g_y = 0.5*(top_left[1] + bottom_right[1])
                goal = tuple([int(centre_g_x), int(centre_g_y)])

    return thymio_centre, dir_vec, angle, thymio_corners, goal

"""
Finds the optimal path given the obstacle contours, the thymio position, the goal, and the size of the thymio in pixels
"""
def live_path(obstacles, thymio_pos, goal, thymio_size, bounds):
    """
    Return the path based on the current thymio position
    """
    polygons = format_polygon_points(obstacles)
    
    path = globnav.get_optimal_path(thymio_size, thymio_pos, goal, polygons, bounds)

    return path

"""
Draws a view of the system on both the image frame and a blank frame
"""
def live_drawing(frame, blank_frame, thymio_corners, obstacles, path):
    if thymio_corners is not None:
        thymio_corners = thymio_corners.reshape((4,2))
        (top_left, top_right, bottom_right, bottom_left) = thymio_corners

        top_right = (int(top_right[0]), int(top_right[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
        top_left = (int(top_left[0]), int(top_left[1]))

        centre_x = 0.5*(top_left[0] + bottom_right[0])
        centre_y = 0.5*(top_left[1] + bottom_right[1])

        centre_thymio = tuple([int(centre_x), int(centre_y)])

        # Draw box around the code in MAIN CAMERA frame
        cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
        cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
        cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
        cv2.line(frame, bottom_left, top_left, (255, 0, 0), 2)

        # Draw box around the thymio in MAP frame
        # cv2.line(blank_frame, top_left, top_right, (0, 255, 0), 2)
        # cv2.line(blank_frame, top_right, bottom_right, (0, 255, 0), 2)
        # cv2.line(blank_frame, bottom_right, bottom_left, (0, 255, 0), 2)
        # cv2.line(blank_frame, bottom_left, top_left, (255, 0, 0), 2)

        # Mark the corners of the shapes in the main frame
        # cv2.putText(frame, 'TL', top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 4)
        # cv2.putText(frame, 'TR', top_right, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 4)
        # cv2.putText(frame, 'BR', bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 4)
        # cv2.putText(frame, 'BL', bottom_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 4)

        dir_x = int(top_left[0] - bottom_left[0])
        dir_y = int(top_left[1] - bottom_left[1])

        dir_thymio = [dir_x, dir_y]
        unit_dir_thymio = dir_thymio / np.linalg.norm(dir_thymio)

        cv2.putText(frame, "direction: {}".format(unit_dir_thymio), centre_thymio, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.line(frame, centre_thymio, tuple([int(top_left[0] + top_right[0])//2, int(top_left[1] + top_right[1])//2]), [0,0,255], 4)

        if path is not None:
            for node in path:
                cv2.circle(frame, node, 5, [255,255,0], 5)

            for i in range(len(path)-1):
                cv2.line(frame, path[i], path[i+1], [255,255,0], 2)

    # Do the drawing
    return 0

def main():
    # Open the default camera
    cam = cv2.VideoCapture(constants.CAMERA_ID)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    cv2.namedWindow("Normal", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Map", cv2.WINDOW_AUTOSIZE)

    cv2.startWindowThread()

    # Warmup of camera
    for i in range(100):
        ret, bin_frame = cam.read()
    
    ret, obstacles_frame = cam.read()

    global_path_found = False

    pos = tuple([0,0])
    path = []
    obstacles = []

    while True:
        ret, frame = cam.read()

        blank_frame = np.zeros(frame.shape)

        pos, _, _, thymio_corners, goal = live_pos(frame, detector, constants.THYMIO_ID, constants.GOAL_ID)

        if pos == None:
            print("Thymio not visible")

        if global_path_found == False:
            obstacles = find_obstacles(obstacles_frame)
            if pos is not None:
                path = live_path(obstacles, pos, goal, constants.THYMIO_SIZE)
                global_path_found = True

        live_drawing(frame, blank_frame, thymio_corners, obstacles, path)

        tl ,tr, bl, br = get_bounds(frame, detector)

        cv2.circle(frame, tl, 3, [255,0,255], 3)
        cv2.circle(frame, tr, 3, [255,0,255], 3)
        cv2.circle(frame, bl, 3, [255,0,255], 3)
        cv2.circle(frame, br, 3, [255,0,255], 3)

        # Display the captured frame
        cv2.imshow("Normal", frame)
        cv2.imshow("Map", blank_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(10) == ord('q'):
            break

    # Release the capture and writer objects
    release_camera(cam)

if __name__ == "__main__":
    main()