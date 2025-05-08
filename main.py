import cv2
import mediapipe as mp
import random
import time

# Initialize hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Choices
choices = ["Rock", "Paper", "Scissors"]

# Game state
player_move = "Waiting..."
computer_move = "..."
result = "..."
last_gesture_time = 0
gesture_cooldown = 2

# Score tracking
player_score = 0
computer_score = 0

def classify_gesture(hand_landmarks):
    tips = [8, 12, 16, 20]
    thumb_tip = 4
    thumb_ip = 2

    fingers = []
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_ip].x:
        fingers.append(1)
    else:
        fingers.append(0)

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

def get_computer_choice():
    return random.choice(choices)

def determine_winner(player, computer):
    if player == computer:
        return "It's a Tie!"
    elif (player == "Rock" and computer == "Scissors") or \
         (player == "Paper" and computer == "Rock") or \
         (player == "Scissors" and computer == "Paper"):
        return "You Win!"
    else:
        return "Computer Wins!"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if current_time - last_gesture_time > gesture_cooldown:
                player_move = classify_gesture(hand_landmarks)
                if player_move != "Unknown":
                    computer_move = get_computer_choice()
                    result = determine_winner(player_move, computer_move)

                    # Update score
                    if "You Win" in result:
                        player_score += 1
                    elif "Computer Wins" in result:
                        computer_score += 1

                    last_gesture_time = current_time
                else:
                    player_move = "Waiting..."
                    computer_move = "..."
                    result = "..."

    # Create grayscale and threshold versions
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # Add game text to frames
    for f in [frame, gray_bgr, thresh_bgr]:
        cv2.putText(f, f"Your Move: {player_move}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(f, f"Computer: {computer_move}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(f, f"Result: {result}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(f, f"Score - You: {player_score}  Computer: {computer_score}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 255), 2)
        if current_time - last_gesture_time < gesture_cooldown:
            countdown = max(0, gesture_cooldown - (current_time - last_gesture_time))
            cv2.putText(f, f"Next try in: {countdown:.1f}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # Show all three views
    cv2.imshow("Original", frame)
    cv2.imshow("Grayscale", gray_bgr)
    cv2.imshow("Threshold", thresh_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
