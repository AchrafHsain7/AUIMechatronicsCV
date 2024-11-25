import cv2 # Used for handling video input and displaying frames.
import mediapipe as mp # Provides hand detection and tracking model

import random
from enum import Enum

class Move(Enum):
  SCISSOR = 'Scissor'
  ROCK = 'Rock'
  PAPER = 'Paper'
  UNKNOWN = 'UNKNOWN'

class GameStatus(Enum):
  DRAW = "It's a draw"
  USER_LOST = "You Lost!"
  USER_WON = "You Won!"


def main():
  mp_hands = mp.solutions.hands
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  capture = cv2.VideoCapture(0)

  with mp_hands.Hands(
      max_num_hands=1,
      model_complexity=0,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands:
    
    # Frame processing loop
    while capture.isOpened():
      success, frame = capture.read() # Capture a single frame from the webcam
      if not success:
        print("Ignoring empty camera frame.")
        continue
      
      game_started = False
      if cv2.waitKey(5) & 0xFF == 32: # 32 == Space
        game_started = True
      elif cv2.waitKey(5) & 0xFF == 27: # 27 == ESC
        break
      
      if not game_started:
        cv2.imshow('Hand Detection', cv2.flip(frame, 1))
        continue
      
      
      
      # To improve performance, optionally mark the frame as not writeable
      frame.flags.writeable = False
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = hands.process(frame)
      
      # undo previous changes
      frame.flags.writeable = True
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

      # if hand landmarks are not detected, return the frame as it is
      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              frame,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
          
          user_move = get_user_move(hand_landmarks)
          if user_move == Move.UNKNOWN.value:
            continue
          computer_move, game_status = get_game_status(user_move)

          print('\n\nYour Move:', user_move)
          print('Computer Move:', computer_move)
          print('Game result:', game_status)
          
      # Flip the frame horizontally for a selfie-view display, then show it
      cv2.imshow('Hand Detection', cv2.flip(frame, 1))

  # freeing any used resources
  capture.release()
  cv2.destroyAllWindows()


def get_finger_status(hand_landmarks):
  finger_status = ''
  finger_types = {'INDEX': 6, 'MIDDLE': 10, 'RING': 14, 'PINKY': 18}
  
  # status of the thump
  thump_mcp_x = hand_landmarks.landmark[2].x
  thump_tip_x = hand_landmarks.landmark[4].x
  if thump_tip_x <= thump_mcp_x:
    finger_status += '0'
  else:
    finger_status += '1'
  
  # status of the rest of 4 fingers
  for finger in finger_types.values():
    finger_pip_y = hand_landmarks.landmark[finger].y
    finger_tip_y = hand_landmarks.landmark[finger+2].y
    
    if finger_tip_y >= finger_pip_y:
      finger_status += '0'
    else:
      finger_status += '1'
  
  return finger_status


def get_user_move(hand_landmarks):
  finger_status = get_finger_status(hand_landmarks)
  match finger_status:
    case '00000':
      return Move.ROCK.value
    case '11111':
      return Move.PAPER.value
    case "01100":
      return Move.SCISSOR.value
    case _:
      return Move.UNKNOWN.value


def get_game_status(user_move):
  scenarios_result = {Move.ROCK.value:Move.SCISSOR.value, Move.PAPER.value:Move.ROCK.value, Move.SCISSOR.value:Move.PAPER.value}
  
  computer_move = random.choice(list(scenarios_result.keys()))
  
  if user_move == computer_move:
    return computer_move, GameStatus.DRAW.value
  elif scenarios_result[user_move] == computer_move:
    return computer_move, GameStatus.USER_WON.value

  return computer_move, GameStatus.USER_LOST.value


if __name__ == '__main__':
  main()