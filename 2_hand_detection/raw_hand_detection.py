import cv2 # Used for handling video input and displaying frames.
import mediapipe as mp # Provides hand detection and tracking model

def main():
    mp_hand = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    capture = cv2.VideoCapture(0)

    with mp_hand.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hand:
    
        # Frame processing loop
        while capture.isOpened():
            # Capture a single frame from the webcam
            success, frame = capture.read() 
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # To improve performance, optionally mark the frame as not writeable
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hand.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # if hand landmarks are not detected, return the frame as it is
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hand.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            # Flip the frame horizontally for a selfie-view display
            cv2.imshow('Hand Detection', cv2.flip(frame, 1))

            # if ESC key pressed, exit the program
            if cv2.waitKey(5) & 0xFF == 27: # 27 == ESC
                break

        # freeing any used resources
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()