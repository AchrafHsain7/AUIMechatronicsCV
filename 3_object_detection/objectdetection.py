import cv2
import os


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    base_path = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_path, "config.pbtxt")
    model_path = os.path.join(base_path, "model.pb")
    labels_path = os.path.join(base_path, "labels.txt")


    net = cv2.dnn_DetectionModel(model_path, cfg_path)
    classes = []
    with open(labels_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]


    net.setInputSize(320, 320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    img = cv2.imread("../images/img2.jpg")
    img = cv2.resize(img, (320, 320))
    classidx, conf, boxes = net.detect(img, confThreshold=0.5)
    for classid, conf, box in zip(classidx, conf, boxes):
        cv2.rectangle(img, box, (255, 0, 0), 1)
        cv2.putText(img, classes[classid], (box[0]+10, box[1]+40) , cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv2.imshow("Image", img)
    

    while True:
        # BASICS 
        ret, frame = cap.read()
        #resizing the frame
        frame = cv2.resize(frame, (320, 320))
        # Text: frame, text, location, font, fontsize, color, fontwidth
        cv2.putText(frame, f"FPS: {cap.get(cv2.CAP_PROP_FPS)}",(10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

        classidx, conf, boxes = net.detect(frame, confThreshold=0.6)

        for classid, conf, box in zip(classidx, conf, boxes):
            cv2.rectangle(frame, box, (255, 0, 0), 1)
            try:
                cv2.putText(frame, classes[classid], (box[0]+10, box[1]+40) , cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            except:
                print(classid)
                print(classes)
        

        frame = cv2.resize(frame, (700, 500))
        cv2.imshow("Frame", frame)


    cap.release()
    cv2.destroyAllWindows()
