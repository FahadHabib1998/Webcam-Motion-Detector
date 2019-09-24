import cv2

initialFrame = None

video = cv2.VideoCapture(0)

while True:
    
    """here check is a boolean, to check if video is running and  frame in a numpy array of the first image the video captures"""
    check, frame = video.read()
    
    """converting frame to grayscale"""
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    """apply galsium blur to the image, to remove noise and improve calculation of the difference"""
    gray = cv2.GaussianBlur(gray,(21,21),0)
    
    if(initialFrame is None):
        initialFrame = gray
        continue
    """difference between the first frame and the current frame"""
    difference = cv2.absdiff(initialFrame,gray)

    """setting any pixel above and equal to 30 to 255(totally white)"""
    thresholdDiff = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)[1]

    """Dilate the black holes from the bigger white areas, smoothing out the white areas"""
    thresholdDiff = cv2.dilate(thresholdDiff, None, iterations = 2)

    """finding contors"""
    (cnts,_) = cv2.findContours(thresholdDiff.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for countours in cnts:
        if cv2.contourArea(countours) < 1000:
            continue
        (x,y,w,b) = cv2.boundingRect(countours)
        cv2.rectangle(frame,(x,y), (x+w,y+b), (0,255,0), 3)
    """Making a window"""
    cv2.imshow("capturing....", gray)
    cv2.imshow("delta...", difference)
    cv2.imshow("threshold....", thresholdDiff)
    cv2.imshow("color frame......", frame)
    
    """Waits for the time passed in the argument when a key is pressed"""
    key = cv2.waitKey(1)
    
    if(key == ord('q')):
        break
    
video.release()
cv2.destroyAllWindows 
