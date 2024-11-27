import cv2
import numpy as np
import matplotlib.pyplot as plt




if __name__ == "__main__":

    img = cv2.imread("cells.jpg")
    cv2.imshow("Cells", img)
    cv2.waitKey(0)

    
    #visualizing the Three Channels over the X axis
    blue = img[:,:,0]
    blue = blue.mean(axis=0)
    green = img[:,:,1]
    green = green.mean(axis=0)
    red = img[:,:,2]
    red = red.mean(axis=0)

    fig, ax = plt.subplots(3, figsize=(10, 5))
    ax[0].plot(blue, c="b")
    ax[1].plot(green, c="g")
    ax[2].plot(red, c="r")
    cv2.imshow("Cell", img)


    # plt.plot(blue)
    plt.show()
    cv2.waitKey(0)

    #Channel Visualization of pixel values
    img_b = img[:, :, 0]
    blue = img_b.reshape((-1))
    img_g = img[:, :, 1]
    green = img_g.reshape((-1))
    img_r = img[:, :, 2]
    red = img_r.reshape((-1))

    plt.hist(blue, bins=100, color='b', alpha=0.3)
    plt.hist(green, bins=100, color='g', alpha=0.3)
    plt.hist(red, bins=100, color='r', alpha=0.3)

    plt.show()

    #Testing the hypothesis
    mask = np.zeros(img.shape[:2])
    h, w, _ = img.shape
    threshold = 140
    print(h, w)
    for i in range(h):
        for j in range(w):
            if img[i, j, 0] < threshold:
                mask[i,j] = 1
    
    plt.imshow(mask)
    plt.show()


    #Decising to use the B channel
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) * mask
    img = img.astype('uint8')
    print(img.mean())
    print(img.shape)
    cv2.imshow("Segmented", img)
    cv2.waitKey(0)

    #Using the Simple Blob detector
    params = cv2.SimpleBlobDetector_Params()
    # params.minThreshold = 160
    params.maxThreshold = 150
    params.filterByArea = True
    params.minArea = 20
    params.filterByCircularity = True
    params.minCircularity = 0.2
    params.filterByConvexity = True
    params.minConvexity = 0.2
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    
    img = cv2.imread("cells.jpg")
    img = img[:,:,0]

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    # print(len(keypoints))
    
    img = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img = cv2.resize(img, (700, 700))
    cv2.imshow("Cells", img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()