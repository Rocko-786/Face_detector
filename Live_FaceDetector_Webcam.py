import cv2
from random import randrange

B_Thick = 3
topBar = 'Face Detector by Rocko'
# print(cv2.__file__)

# Loading the trained face data into the program
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Loading the image into the program
# img = cv2.imread('group.jpeg')
webcam = cv2.VideoCapture(0)

while True:
    flag, img_frame = webcam.read()
    B_Color = (randrange(256), randrange(256), randrange(256))
    # if flag:
    # Gray Scaling the image cause we only need to detect thow to kill a running webcam running in background in linuxhe face
    # Which can be done in grayscale also 
    # so to maintain the complexity we are doing this
    grayscaled_img = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)

    # Detecting Faces & Finding the co-ordinates of the image
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    for (x, y, w, h) in face_coordinates:
        upper_left = (x, y)
        bottom_right = (x+w, y+h)

        # Drawing the rectangles
        cv2.rectangle(img_frame, upper_left, bottom_right, B_Color, B_Thick)


    # This will show the image with title
    cv2.imshow(topBar, img_frame)
    key = cv2.waitKey(1)

    if key == 27 or key == 81 or key == 113 or cv2.getWindowProperty(topBar, cv2.WND_PROP_VISIBLE) < 1:
        break


# Release the videoCapture object(webcam)
webcam.release()