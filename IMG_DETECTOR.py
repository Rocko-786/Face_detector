import cv2

B_Color = (0, 255, 0)
B_Thick = 2


# Loading the trained face data into the program
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Loading the image into the program
img = cv2.imread('group.jpeg')

# Gray Scaling the image cause we only need to detect the face
# Which can be done in grayscale also 
# so to maintain the complexity we are doing this
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detecting Faces & Finding the co-ordinates of the image
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
for (x, y, w, h) in face_coordinates:
    upper_left = (x, y)
    bottom_right = (x+w, y+h)

    # Drawing the rectangles
    cv2.rectangle(img, upper_left, bottom_right, B_Color, B_Thick)


# This will show the image with title
cv2.imshow('Face Detector by Rocko', img)

# This let the program pause untill you press any key
cv2.waitKey()