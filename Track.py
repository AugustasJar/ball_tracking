from imutils.video import VideoStream
import numpy as np
import cv2 #pip install opencv-contrib-python
import imutils
import time
import math
from shapely.geometry import Point, Polygon

img_width = 720
img_height = 480
base_width = 720
fogOfWar = 125
theta = (30*math.pi)/180
def videoLoop(upperB,lowerB):
    vs = VideoStream(0)
    vs.start()
    time.sleep(1.0)
    while True:
        # grab the current frame
        frame = vs.read()
        if frame is None:
            break

        frame = cv2.resize(frame, (img_width,img_height))
        mask = create_mask(frame,lowerB,upperB)
        cnts = findCountours(mask)
        frame = displayDetection(frame,cnts)

        # angle could be more useful when inputing controls, pts - points are the quadrilateral corners.
        angle,pts = findBestPath(cnts)

        if angle:
            frame = drawLine(frame,pts[0],pts[1])
            frame = drawLine(frame,pts[2],pts[3])
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(100) & 0xFF
        if key == ord("q"):
            break
    vs.stop()
    cv2.destroyAllWindows()

#proprecesses the image based on a hsv range (LowerBounds, upperBounds)
def create_mask(image,LB,UB):
    hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv_image, (7,7),0)
    LB = np.array(LB)
    UB = np.array(UB)
    mask = cv2.inRange(blurred,LB,UB)
    mask = mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask

#dumb contour detection, need to add template matching or something.
def findCountours(mask):
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

# find the largest contour in the mask, then use
# it to compute the minimum enclosing circle and
# centroid
def displayDetection(frame,cnts):
    if len(cnts) > 0:

        center = None
        for c in cnts:
            if (cv2.contourArea(c) > 1000):

                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
    return frame


def drawLine(frame,p1,p2):

    cv2.line(frame,p1,p2,(0,0,255),2)
    return frame

#generates a quadrilateral based on parameters
# shape parameters are base_width, fogOfWar (range) and theta (default angle of arms)
def generatePath(angle):
    angle = math.pi*angle/180
    x0 = math.floor((img_width - base_width) / 2)
    y0 = img_height
    x1 = x0 + math.floor(img_height * math.tan(angle+theta))
    y1 = fogOfWar
    x2 = img_width - x0
    y2 = y0
    x3 = x2 - math.floor(img_height * math.tan(-angle+theta))
    y3 = y1
    return [(x0,y0),(x1,y1),(x2,y2),(x3,y3)]

#sweeps the screen in a quadrilateral shape
#
#chooses the angle with the highest concentration of contours.
# to improve - scan from left to center, then from right to center, because now it has a right bias
# use countour area as a weight to decide which balls are closer - easier to get.
# if no balls are found decrease fogOfWar to see further.
def findBestPath(cnts):
    maxN = 0
    N = 0
    bestAngle = None
    bestPath = None
    for angle in range(-25,26,5):
        pts = generatePath(angle)
        poly = Polygon(pts)
        for c in cnts:
            p = Point(contourCenter(c))
            if (poly.contains(p)):
                N+=1
        if (N > maxN):
            bestAngle = angle
            bestPath = pts
            maxN=N
        N=0
    return bestAngle,bestPath
#same math as in shapely library, is wonky, not gonna use it.
def complimentary(A,B,C,P):
    v0 = np.subtract(C,A)
    v1 = np.subtract(B,A)
    v2 = np.subtract(P,A)
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    return (u > 0) and (v > 0) and (u + v < 1)


def contourCenter(c):
   # ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return center


def lineLength(p1,p2):
    return math.sqrt(pow(p1[0] - p2[0],2) + pow(p1[1]-p2[1],2))


def TriangleArea(a,b,c):
    p = (a + b + c)/2
    A = math.sqrt(p*(p-a)*(p-b)*(p-c))
    return A