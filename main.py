import cv2
import mediapipe as mp
import random
import time
import os
from datetime import datetime

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Game data
choices = ["Rock", "Paper", "Scissors"]
player_score = 0
computer_score = 0
gesture_cooldown = 3
last_gesture_time = 0
player_move = "Waiting..."
computer_move = "..."
result = "..."

# Gesture classification
def classify_gesture(hand_landmarks):
    tips = [8, 12, 16, 20]
    thumb_tip = 4
    thumb_ip = 2
    fingers = []

    # Thumb
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_ip].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Fingers
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    if sum(fingers) == 0:
        return "Rock"
    elif sum(fingers) == 5:
        return "Paper"
    elif fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
        return "Scissors"
    else:
        return "Unknown"

def determine_winner(player, computer):
    if player == computer:
        return "It's a Tie!"
    elif (player == "Rock" and computer == "Scissors") or \
         (player == "Paper" and computer == "Rock") or \
         (player == "Scissors" and computer == "Paper"):
        return "You Win!"
    else:
        return "Computer Wins!"

# Add game text to image
def add_text(img):
    cv2.putText(img, f"Move: {player_move}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, f"Computer: {computer_move}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(img, f"Result: {result}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, f"Score You: {player_score} | CPU: {computer_score}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
    if time.time() - last_gesture_time < gesture_cooldown:
        wait = gesture_cooldown - (time.time() - last_gesture_time)
        cv2.putText(img, f"Next in: {wait:.1f}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if time.time() - last_gesture_time > gesture_cooldown:
                player_move = classify_gesture(hand_landmarks)
                if player_move != "Unknown":
                    computer_move = random.choice(choices)
                    result = determine_winner(player_move, computer_move)

                    # Update score
                    if "You Win" in result:
                        player_score += 1
                    elif "Computer Wins" in result:
                        computer_score += 1

                    # Save screenshots
                    if not os.path.exists("screenshots"):
                        os.makedirs("screenshots")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"screenshots/original_{timestamp}.png", frame)
                    cv2.imwrite(f"screenshots/grayscale_{timestamp}.png", gray_bgr)
                    cv2.imwrite(f"screenshots/threshold_{timestamp}.png", thresh_bgr)
                    cv2.imwrite(f"screenshots/binarized_{timestamp}.png", binarized_bgr)

                    last_gesture_time = time.time()
                else:
                    player_move = "Waiting..."
                    computer_move = "..."
                    result = "..."

    # Image processing steps
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    binarized = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

    # Convert to BGR for display
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    binarized_bgr = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)

    # Add text overlays
    frame = add_text(frame)
    gray_bgr = add_text(gray_bgr)
    thresh_bgr = add_text(thresh_bgr)
    binarized_bgr = add_text(binarized_bgr)

    # Display all views
    cv2.imshow("Original", frame)
    cv2.imshow("Grayscale", gray_bgr)
    cv2.imshow("Threshold", thresh_bgr)
    cv2.imshow("Binarization", binarized_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
