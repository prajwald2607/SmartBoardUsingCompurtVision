import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

def main():
    st.title("Hand Tracking Paint App")

    # Create a canvas
    canvas = st.empty()
    canvas_width, canvas_height = 640, 480
    canvas_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Initialize hand tracking
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    while True:
        # Get webcam frame
        _, frame = st.beta_read_all()

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get hand landmarks
        results = hands.process(rgb_frame)

        # Draw hand landmarks on the canvas
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * canvas_width), int(landmark.y * canvas_height)
                    cv2.circle(canvas_image, (x, y), 10, (255, 0, 0), -1)

        # Display the canvas image
        canvas.image(canvas_image, channels="BGR", use_column_width=True)

if __name__ == "__main__":
    main()
