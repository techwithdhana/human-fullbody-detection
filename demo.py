import cv2

pedestrian_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Function to perform pedestrian detection from images. Pass an image as a variable.
def humanBodyDetection(frame):

    pedestrians = pedestrian_cascade.detectMultiScale( frame, 1.1, 1)
    # To draw a rectangle on each pedestrians
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, 'Person', (x + 6, y - 6), font, 0.5, (0, 255, 0), 1)
        # Display frames in a window
    return frame

# read image:
# cap = cv2.VideoCapture('image.jpg')

# _, img =  cap.read()
# img = humanBodyDetection(img)
# cv2.imshow('image', img)
# cv2.waitKey(0)

# read video:
cap = cv2.VideoCapture('people.mp4')

while True:
	_, frame = cap.read()
	video = humanBodyDetection(frame)

	cv2.imshow('video', video)
	if cv2.waitKey(10) & 0xFF == ord('s'):
		break

cap.release()
cv2.destroyAllWindows()