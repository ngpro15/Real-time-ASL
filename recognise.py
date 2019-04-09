import cv2
import numpy as np
import imutils


def nothing(x):
    pass

image_x, image_y = 64,64

import h5py
from keras.models import load_model
classifier = load_model('Trained_model.h5')           


def predictor():
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       
       if result[0][0] == 1:
              return 'A'
       elif result[0][1] == 1:
              return 'B'
       elif result[0][2] == 1:
              return 'C'
       elif result[0][3] == 1:
              return 'D'
       elif result[0][4] == 1:
              return 'E'
       elif result[0][5] == 1:
              return 'F'
       elif result[0][6] == 1:
              return 'G'
       elif result[0][7] == 1:
              return 'H'
       elif result[0][8] == 1:
              return 'I'
       elif result[0][9] == 1:
              return 'J'
       elif result[0][10] == 1:
              return 'K'
       elif result[0][11] == 1:
              return 'L'
       elif result[0][12] == 1:
              return 'M'
       elif result[0][13] == 1:
              return 'N'
       elif result[0][14] == 1:
              return 'O'
       elif result[0][15] == 1:
              return 'P'
       elif result[0][16] == 1:
              return 'Q'
       elif result[0][17] == 1:
              return 'R'
       elif result[0][18] == 1:
              return 'S'
       elif result[0][19] == 1:
              return 'T'
       elif result[0][20] == 1:
              return 'U'
       elif result[0][21] == 1:
              return 'V'
       elif result[0][22] == 1:
              return 'W'
       elif result[0][23] == 1:
              return 'X'
       elif result[0][24] == 1:
              return 'Y'
       elif result[0][25] == 1:
              return 'Z'
       

       
bg = None
def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight) #The function calculates the weighted sum of the input 'image' and the accumulator 'bg' so that 'bg' becomes a running average of a frame sequence


def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff,threshold,255,cv2.THRESH_BINARY)[1]
    (_, cnts, _) = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #when no contours detected
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


if __name__ == "__main__":

    aWeight = 0.5
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    img_text = ''

    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0

    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame,1)
        frame2=imutils.resize(frame, width=700)
        #frame2 = cv2.flip(frame2, 1)
        clone = frame2.copy()

        (height, width) = frame2.shape[:2]
        roi = frame2[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 30:         #Until we get past 30 frames, we keep on adding the input frame to our run_avg function and update our background model
            run_avg(gray, aWeight)

        else:
            hand = segment(gray)
            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)


            cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
       
    
            cv2.putText(clone, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
            cv2.imshow("test", clone)
    
        #if cv2.waitKey(1) == ord('c'):
        
            img_name = "1.png"
            if hand is not None:
                save_img = cv2.resize(thresholded, (image_x, image_y))
                cv2.imwrite(img_name, save_img)
                img_text = predictor()


        num_frames += 1
        

        if cv2.waitKey(1) == 27:
            break


    cam.release()
    cv2.destroyAllWindows()