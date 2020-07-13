import numpy as np
import cv2
import sys
import autopy

# Resolution of display
screen_size = autopy.screen.size()
screen_size = (int(screen_size[0]), int(screen_size[1]))

cap = cv2.VideoCapture(0)
# Resolution of webcam frame capture
cap_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

background_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=False)

# Default values for sensitive box, covers 0.25 of width and 0.5 of height
roi_x = 0.5
roi_y = 0.5

if(len(sys.argv) != 1 and len(sys.argv) != 2 and len(sys.argv) != 3):
    print('Usage: python3 main.py [roi_x [roi_y]]')
    print('roi_x and roi_y define the dimensions of the region of interest/box (i.e., sensitive to finger detection).')
    print('Default values of roi_x and roi_y are 0.5 and 0.5, which means the box covers quarter the width and half the height.')
    sys.exit()

if(len(sys.argv) == 2):
    if(float(sys.argv[1]) < 0 or float(sys.argv[1]) > 1):
        print('roi_x and roi_y must be in the range [0, 1].')
        sys.exit()
    roi_x = float(sys.argv[1])

if(len(sys.argv) == 3):
    if(float(sys.argv[1]) < 0 or float(sys.argv[1]) > 1 or float(sys.argv[2]) < 0 or float(sys.argv[2]) > 1):
        print('roi_x and roi_y must be in the range [0, 1].')
        sys.exit()
    roi_x = float(sys.argv[1])
    roi_y = float(sys.argv[2])

# Resolution of region of interest
roi_size = (int(roi_x * cap_size[0]), int(roi_y * cap_size[1]))

# Need to map roi_size to screen_size via x and y scale factors
x_scale_factor = screen_size[0] / roi_size[0]   # Motion by 1 unit along roi x-axis is equivalent to motion by x_scale_factor units along screen x-axis
y_scale_factor = screen_size[1] / roi_size[1]    # Motion by 1 unit along roi y-axis is equivalent to motion by y_scale_factor units along screen y-axis

print('Press b to reset background')
print('Press q to quit')
while(True):
    # Reading frame from webcam
    ret, frame = cap.read()

    if(ret == False):
        sys.exit()

    # Defining the region of interest (sensitive to finger detection) based on roi_x and roi_y
    roi = frame[:int(roi_y * frame.shape[0]), :int(roi_x * frame.shape[1])].copy()
    cv2.rectangle(frame, (0, 0), (int(roi_x * frame.shape[1]), int(roi_y * frame.shape[0])), (255, 0, 0), 2)

    # Black and white image with only hand in white
    fingers = background_subtractor.apply(roi)
    # Blurring image to remove noise
    fingers = cv2.medianBlur(fingers, 5)
    # Dilating image to obtain clarity
    fingers = cv2.dilate(fingers, (5, 5), iterations=100)

    # Detecting contours in black and white image of hand
    contours, _ = cv2.findContours(fingers, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours) != 0):
        # Finding the contour with the maximum area
        max_contour = max(contours, key=cv2.contourArea)

        # Finding the convex hull around the contour
        convex_hull = cv2.convexHull(max_contour, False)
        # Drawing the convex hull
        cv2.drawContours(roi, [convex_hull], 0, (0, 255, 0), 2)

        # Finding centroid of convex hull
        moments = cv2.moments(convex_hull)
        if(moments['m00'] != 0):
            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])
        else:
            centroid_x, centroid_y = 0, 0
        # Plotting centroid
        cv2.circle(roi, (centroid_x, centroid_y), 10, (0, 255, 255), -1)

        # Next step is to extract the vertices from approx_polygon because for
        # some reason cv2 represents polygon vertices as [[x, y]] (note the extra square brackets)
        vertices = np.array([vertex[0] for vertex in max_contour])
        # Extracting the topmost vertex
        max_vertex = vertices[np.argmin(vertices[:, 1])]
        cv2.circle(roi, tuple(max_vertex), 10, (0, 0, 255), -1)
        cv2.line(roi, (centroid_x, centroid_y), tuple(max_vertex), (0, 0, 0), 2)

    frame = cv2.flip(frame, 1)
    roi = cv2.flip(roi, 1)
    # cv2.imshow('Frame', frame)
    x = int(autopy.mouse.location()[0])
    y = int(autopy.mouse.location()[1])
    cv2.putText(roi, f'({x}, {y})', (roi_size[0] - max_vertex[0] - 1, max_vertex[1]), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow('ROI', roi)
    # cv2.imshow('Mask', fingers)

    autopy.mouse.smooth_move(screen_size[0] - int(x_scale_factor * max_vertex[0]) - 1, int(y_scale_factor * max_vertex[1]))

    key = cv2.waitKey(1)
    if(key == ord('b') or key == ord('B')):
        background_subtractor = cv2.createBackgroundSubtractorKNN()
    if(key == ord('q') or key == ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()
