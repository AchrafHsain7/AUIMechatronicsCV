import cv2
import numpy as np


img = cv2.imread("../images/img7.jpg")
cv2.imshow("Frame", img)
cv2.waitKey(0)

#graysacle image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Frame", gray)
cv2.waitKey(0)

#COMPUTING
_, tresh = cv2.threshold(gray, np.mean(gray)*0.7, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Frame", tresh)
cv2.waitKey(0)


contours, hiearchy = cv2.findContours(tresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
#sorting nd getting the biggest contour
largest_ctr = sorted(contours, key=cv2.contourArea)[-1]

#creating a mask to keep only the biggets contour
mask = np.zeros(img.shape[:2], dtype="uint8")
maskimg = cv2.drawContours(mask, [largest_ctr], -1, (255, 255, 255), -1)
cv2.imshow("Frame", maskimg)
cv2.waitKey(0)

segmented_img = cv2.bitwise_and(img, img,  mask=maskimg)
cv2.imshow("Frame", segmented_img)
cv2.waitKey(0)



cap = cv2.VideoCapture(0)
while True:
        # BASICS 
        ret, frame = cap.read()
        #resizing the frame
        frame = cv2.resize(frame, (700, 500))
        # Text: frame, text, location, font, fontsize, color, fontwidth
        cv2.putText(frame, f"FPS: {cap.get(cv2.CAP_PROP_FPS)}",(10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, tresh = cv2.threshold(gray, np.mean(gray)*1.1, 255, cv2.THRESH_BINARY_INV)
        contours, hiearchy = cv2.findContours(tresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        largest_ctr = sorted(contours, key=cv2.contourArea)[-1]
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        maskimg = cv2.drawContours(mask, [largest_ctr], -1, (255, 255, 255), -1)
        segmented_img = cv2.bitwise_and(frame, frame,  mask=maskimg)
        cv2.imshow("Window", segmented_img)





cv2.destroyAllWindows()