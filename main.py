import cv2
import time


def get_images():

    # On my laptop
    # 0 front camera
    # 1 back camera
    camera = 0
    cap = cv2.VideoCapture(camera)

    # Capture X number of images
    for x in range(10):
        # take image
        ret, frame = cap.read()

        # display image
        cv2.imshow('frame', frame)

        # write image to file
        #out = cv2.imwrite('./capture' + str(x) + '.jpg', frame)
        cv2.waitKey(1)

        time.sleep(3)

    cap.release()


get_images()
