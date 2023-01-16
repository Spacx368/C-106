import cv2

img = ("walking.avi")
# Create our body classifier

body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # Pass frame to our body classifier
    
    bodies = body_classifier.detectMultiScale(gray_img, 1, 2, 3)
    
    # Extract bounding boxes for any bodies identified
    
    cv2.imshow()

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
