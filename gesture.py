import cv2
import numpy as np
import math

camera = cv2.VideoCapture(0 , cv2.CAP_DSHOW)#cv2.CAP_DSHOW
file = open("data.txt", "w+")
while True:
    try:
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        kernel = np.ones((3, 3), np.uint8)
        region_of_image = frame[0:250, 0:250]

        cv2.rectangle(frame, (0, 0), (250, 250), (255, 0, 0), 0)
        hsv_color = cv2.cvtColor(region_of_image, cv2.COLOR_BGR2HSV)

        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv_color, lower_skin, upper_skin)

        mask = cv2.dilate(mask, kernel, iterations=4)

        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        hull = cv2.convexHull(cnt)

        area_of_hull = cv2.contourArea(hull)
        area_of_hand = cv2.contourArea(cnt)

        arearatio = ((area_of_hull - area_of_hand) / area_of_hand) * 100

        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)

        #no defect
        c = 0

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
                c += 1
                cv2.circle(region_of_image, far, 3, [255, 0, 0], -1)

            cv2.line(region_of_image, start, end, [0, 255, 0], 2)

        c += 1
        arr = np.array([] , dtype=np.str)

        font = cv2.FONT_HERSHEY_SIMPLEX
        if c == 1:
            if area_of_hand < 2000:
                cv2.putText(frame, 'Put hand in the box', (0, 250), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                if arearatio < 12:
                    cv2.putText(frame, 'COME', (0, 250), font, 2, (0, 0, 255), 3, cv2.LINE_AA)#0
                    arr = np.append(arr, ['COME'])
                    np.savetxt('data.txt', [arr], fmt="%s")
                elif arearatio < 17.5:
                    cv2.putText(frame, 'COME', (0, 250), font, 2, (0, 0, 255), 3, cv2.LINE_AA)#best of luck
                    arr = np.append(arr, ['COME'])
                    np.savetxt('data.txt', [arr], fmt="%s")
        elif c>2 & c<=6:
            cv2.putText(frame, 'GO', (0, 250), font, 2, (0, 0, 255), 3, cv2.LINE_AA)  # 5
            arr = np.append(arr, ['GO'])
            np.savetxt('data.txt', [arr], fmt="%s")
    except:
        a = 10

    cv2.imshow('mask', mask)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()
camera.release()


