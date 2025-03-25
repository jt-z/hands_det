import cv2
import numpy as np
from scipy.spatial import distance 

import cv2
import mediapipe as mp
 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

 
 
def extract_landmarks(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.1) as hands:  
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        landmarks_list = []   
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                landmarks_list.append(landmarks)  
        return landmarks_list if landmarks_list else None
    

dog_landmarks_template = np.load('../assets/dog_landmarks.npy', allow_pickle=True)
rabbit_landmarks_template = np.load('../assets/rabbit_landmarks.npy', allow_pickle=True)

def compare_landmarks(landmarks, template_landmarks):
    if len(landmarks) != len(template_landmarks):
        return float('inf')  
    return np.mean([distance.euclidean(landmark, template) for landmark, template in zip(landmarks, template_landmarks)])

def compare_two_hands(landmarks, template):
    if len(landmarks) != 2 or len(template) != 2:
        return float('inf') 
    hand1_similarity = compare_landmarks(landmarks[0], template[0])
    hand2_similarity = compare_landmarks(landmarks[1], template[1])
    return (hand1_similarity + hand2_similarity) / 2 

def hands_distance(landmarks):
    hand1_center = landmarks[0][0] 
    hand2_center = landmarks[1][0] 
    return distance.euclidean(hand1_center, hand2_center)

def detect_hand_shadow(current_landmarks, distance_threshold=0.2):
    hands_dist = hands_distance(current_landmarks)
    
    if hands_dist > distance_threshold:
        return 'Hands too far apart, closeer!'
    
    dog_similarity = compare_two_hands(current_landmarks, dog_landmarks_template)
    
    rabbit_similarity = compare_two_hands(current_landmarks, rabbit_landmarks_template)
    
    if dog_similarity < rabbit_similarity:
        return 'bird Hand Shadow'
    else:
        return 'wolf Hand Shadow'

def main():

    video_file = '../assets/wolf_and_chicken.mp4' 
    video_file = '../assets/self_get.mov'

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(video_file)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter('output_video2.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))
    
    
    distance_threshold = 0.2 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
    
        current_landmarks = extract_landmarks(frame)
        
        if current_landmarks and len(current_landmarks) == 2:
           
            label = detect_hand_shadow(current_landmarks, distance_threshold)
            
           
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
           
            cv2.putText(frame, 'Make sure has two hands.', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        
        cv2.imshow("Hand Shadow Detection", frame)

                
        
        out.write(frame)
        
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
