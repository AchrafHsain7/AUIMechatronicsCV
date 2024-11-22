# Used for handling video input and displaying frames.
import cv2
# Provides the pre-trained hand detection and tracking model
import mediapipe as mp

# Offers tools for hand landmark detection and tracking
mp_hands = mp.solutions.hands
# Used for drawing landmarks and connections on detected hands.
mp_drawing = mp.solutions.drawing_utils
# Contains predefined styles for hand landmarks and connections.
# i.e how landmarks will look
mp_drawing_styles = mp.solutions.drawing_styles
# Initialize a Webcam object that will be used to opens the webcam
# index 0 refers to the default webcam
capture = cv2.VideoCapture(0)

# initialize the hand tracking object, and it will be gone at the end of with
# statement
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  
  # Frame processing loop
  while capture.isOpened():
    success, image = capture.read() # Capture a single frame from the webcam
    if not success:
      print("Ignoring empty camera frame.")
      continue

    # Preparing the image
    #--------------------------------------------------------------------
    # To improve performance, optionally mark the image as not writeable
    image.flags.writeable = False
    
    # Converts the image from BGR (used by OpenCV) to RGB (expected by 
    # Mediapipe)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Processes the image and detects hand landmarks.
    results = hands.process(image)


    # Draw the hand annotations on the image.
    #--------------------------------------------------------------------
    # undo previous changes
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # if hand landmarks are detected, draw them in the frame
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    
    # Flip the image horizontally for a selfie-view display, then show it
    cv2.imshow('Hand Detection', cv2.flip(image, 1))
    
    # if ESC key is pressed, close the webcam
    if cv2.waitKey(5) & 0xFF == 27:
      break

# making sure the used resources by Webcam object are released
capture.release()