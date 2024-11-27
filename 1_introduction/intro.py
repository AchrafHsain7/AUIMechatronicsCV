import cv2
import numpy as np


if __name__ == "__main__":

    # reading an imagels
    img = cv2.imread("../images/img1.jpg")
    cv2.imshow("Image", img)
    #waiting for a key to be pressed: 0 infinitely, other ms?
    cv2.waitKey(0)
    

    # #resizing the image
    img_res = cv2.resize(img, (1920, 1080))
    cv2.imshow("Resize", img_res)
    cv2.waitKey(0)

    #flipping the image
    img_flip = cv2.flip(img, 1) #0 for horizentally, 1 for vertically
    cv2.imshow("flip", img_flip)
    cv2.waitKey(0)

    # cropping the image
    img_crop = img[0:300, 100:300, :]
    cv2.imshow("crop", img_crop)
    cv2.waitKey(0)

    # seeing the channels
    img_np = np.array(img)
    print(img_np.shape)

    blue_channel = img[:, :, 0]  # Extract the Blue channel
    green_channel = img[:, :, 1]  # Extract the Green channel
    red_channel = img[:, :, 2]  # Extract the Red channel

    # Create blank images to fill other channels with 0
    blue_image = np.zeros_like(img)
    green_image = np.zeros_like(img)
    red_image = np.zeros_like(img)

    # Assign the respective channel values
    blue_image[:, :, 0] = blue_channel  
    green_image[:, :, 1] = green_channel  
    red_image[:, :, 2] = red_channel  
    
    cv2.imshow("Blue", blue_image)
    cv2.imshow("Green", green_image)
    cv2.imshow("Red", red_image)
    cv2.waitKey(0)

    combined = blue_image + green_image + red_image
    cv2.imshow("Combined", combined)
    cv2.waitKey(0)

    # getting the grayscale image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray_img)
    

    # blurring an image
    blur_img = cv2.GaussianBlur(img, (3, 3),cv2.BORDER_DEFAULT )
    cv2.imshow("Blured", blur_img)
    cv2.waitKey(0)  

    # creating a video
    cap = cv2.VideoCapture("../images/vid1.mp4")

    while True:

        ret, frame = cap.read()
        if ret == False:
            break

        frame = cv2.resize(frame, (320, 320))
        cv2.imshow("Video", frame)

        if cv2.waitKey(20) & 0xFF==ord('q'): #checking if the keyboard q was pressed
            break

    #Live Camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)

        #adding text to an image
        cv2.putText(frame, f"{cap.get(cv2.CAP_PROP_FPS)}", (50, 50), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255), thickness=1)

        #adding a rectangle
        cv2.rectangle(frame, (100, 50), (500, 400), (255, 0, 0), 2, lineType=cv2.LINE_AA) #Thickness -1 for filled

        #drawing a circle
        cv2.circle(frame, (300, 300), 100, (0,0,255), 2) 

        #drawing a line
        cv2.line(frame, (10, 10), (400, 300), (0,255,0), 3)

        frame = cv2.Canny(frame, 125, 175)


        cv2.imshow("Camera", frame)

        if cv2.waitKey(20) & 0xFF==ord('q'): #checking if the keyboard q was pressed
            break



    cap.release()  #releasing the capture
    cv2.destroyAllWindows()


