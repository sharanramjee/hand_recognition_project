import cv2
import time
import platform

IS_MAC = (platform.system() == 'Darwin')

def get_images(numSamples: int, secsBetweenSamples: float):

    # On my laptop
    # 0 front camera
    # 1 back camera
    camera = 0
    cap = cv2.VideoCapture(camera)

    # Capture X number of images
    for x in range(numSamples):
        # take image
        ret, frame = cap.read()

        # display image
        cv2.imshow('frame', frame)

        #   Mac default camera is 1280 by 720 (16:9)
        if (IS_MAC):
            frame = frame[:, 160:1120, :]   #   Extract the middle column of the image (go from 16:9 resolution to 4:3 resolution)

        #   Ensure common resolution
        frame = cv2.resize(frame, (640, 480))

        # write image to file
        if (not IS_MAC or x):   #   The first image doesn't save on mac
            out = cv2.imwrite('./capture{}.jpg'.format(x), frame)
        cv2.waitKey(1)

        time.sleep(secsBetweenSamples)

    cap.release()

if __name__ == '__main__':
    get_images(10, 1)
