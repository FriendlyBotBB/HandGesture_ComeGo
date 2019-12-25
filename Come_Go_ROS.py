import cv2
import numpy as np
import math
import rospy
from  geometry_msgs.msg import Twist

rospy.init_node('burak_bilge_node')

publisher_turtle1 = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size = 1)

movement = Twist()

movement.linear.x = 0.0
movement.linear.y = 0.0
movement.linear.z = 0.0
movement.angular.x = 0.0
movement.angular.y = 0.0
movement.angular.z = 0.0

cap = cv2.VideoCapture(0)

while (1):
    try:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        kernel = np.ones((3, 3), np.uint8)

        roi = frame[0:250, 0:250]

        cv2.rectangle(frame, (0, 0), (250, 250), (0, 255, 0), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        mask = cv2.dilate(mask, kernel, iterations=4)

        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        hull = cv2.convexHull(cnt)

        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)

        arearatio = ((areahull - areacnt) / areacnt) * 100

        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)

        l = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt = (100, 180)

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

            d = (2 * ar) / a

            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            if angle <= 90 and d > 30:
                l += 1
                cv2.circle(roi, far, 3, [255, 0, 0], -1)

            cv2.line(roi, start, end, [0, 255, 0], 2)

        l += 1

        font = cv2.FONT_HERSHEY_SIMPLEX
        if l == 1:
            if areacnt < 2000:
                cv2.putText(frame, 'Put hand in the box', (0, 250), font, 2, (0, 0, 255), 3, 16L)
                movement.linear.x = 0
            else:
                if arearatio < 12:
                    cv2.putText(frame, 'COME', (0, 250), font, 2, (0, 0, 255), 3, 16L)
                    movement.linear.x = 0.5
                elif arearatio < 17.5:
                    cv2.putText(frame, 'COME', (0, 250), font, 2, (0, 0, 255), 3, 16L)
                    movement.linear.x = 0.5
                else:
                    cv2.putText(frame, '1', (0, 250), font, 2, (0, 0, 255), 3, 16L)

        elif l >= 2 & l <=6:
            cv2.putText(frame, 'GO', (0, 250), font, 2, (0, 0, 255), 3, 16L)
            movement.linear.x = -0.5


    except:
        a=10

    cv2.imshow('mask', mask)
    cv2.imshow('frame', frame)

    publisher_turtle1.publish(movement)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

