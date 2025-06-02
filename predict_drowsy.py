import cv2
import time
import numpy as np
import winsound
from collections import deque

from tensorflow.keras.models import load_model
from scipy.signal import detrend
from scipy.fftpack import fft

# Load models
eye_model = load_model("eye_model.h5")
yawn_model = load_model("yawn_model.h5")

# Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mcs_mouth.xml")

# Constants
EYE_CLOSED_ALARM_DURATION = 5
FACE_MISSING_THRESHOLD = 5
NO_EYE_THRESHOLD = 5
BLINK_WINDOW = 10
BLINK_THRESHOLD = 2

# Initial flags and counters
previous_eye_status = None
blink_count = 0
blink_timer_start = time.time()
last_face_time = time.time()
eye_closed_start_time = None
no_eye_detected_time = None
last_eye_status = "open"
blink_times = deque()
last_blink_alert_time = 0
last_beep_time = 0
green_signal = deque(maxlen=300)  # ~10 seconds at 30 FPS
heart_rate = 0
last_hr_update = time.time()
yawn_frame_counter = 0
yawn_yawn_counter = 0
yawn_timer_start = time.time()
cap = cv2.VideoCapture(0)

# Helper function
def play_beep():
    winsound.Beep(1000, 500)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.1, 5)

    eye_status = "Detecting..."
    yawn_status = "Detecting..."
    face_missing_flag = False
    no_eye_flag = False
    eye_closed_flag = False
    yawn_flag = False
    blink_rate_flag = False

    now = time.time()
    current_time = time.time()
    elapsed_time = current_time - blink_timer_start

    if len(faces) > 0:
        last_face_time = now

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 200), 2)
            roi_gray = frame_gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Forehead region for heart rate
            forehead = frame[y:y + int(h * 0.15), x + int(w * 0.3):x + int(w * 0.7)]
            if forehead.size > 0:
                green_mean = np.mean(forehead[:, :, 1])  # Green channel
                green_signal.append(green_mean)

            # Eye detection
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
            eyes_closed = 0

            if len(eyes) > 0:
                no_eye_detected_time = None
                for (ex, ey, ew, eh) in eyes[:2]:
                    eye_img = roi_gray[ey:ey + eh, ex:ex + ew]
                    try:
                        eye_resized = cv2.resize(eye_img, (64, 64))
                        eye_normalized = eye_resized.astype('float32') / 255.0
                        eye_input = np.expand_dims(np.expand_dims(eye_normalized, axis=-1), axis=0)
                        prediction = eye_model.predict(eye_input, verbose=0)
                        label = "open" if prediction[0][0] > 0.5 else "closed"

                        if label == "open":
                            if last_eye_status == "closed":
                                blink_times.append(now)
                            eye_closed_start_time = None
                            last_eye_status = "open"
                        else:
                            eyes_closed += 1
                            if eye_closed_start_time is None:
                                eye_closed_start_time = now
                            last_eye_status = "closed"

                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                        cv2.putText(roi_color, label, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    except Exception as e:
                        print("Eye error:", e)

                if eyes_closed == len(eyes[:2]):
                    eye_status = "closed"
                else:
                    eye_status = "open"
            else:
                eye_status = "Closed"
                if no_eye_detected_time is None:
                    no_eye_detected_time = now
                if eye_closed_start_time is None:
                    eye_closed_start_time = now

            # Yawn detection
            mouth_roi_gray = roi_gray[h // 2:, :]
            mouths = mouth_cascade.detectMultiScale(mouth_roi_gray, 1.3, 15)
            if len(mouths) > 0:
                for (mx, my, mw, mh) in mouths[:1]:
                    try:
                        mouth_img = mouth_roi_gray[my:my + mh, mx:mx + mw]
                        mouth_resized = cv2.resize(mouth_img, (64, 64))
                        mouth_normalized = mouth_resized.astype('float32') / 255.0
                        mouth_input = np.expand_dims(np.expand_dims(mouth_normalized, axis=-1), axis=0)
                        yawn_pred = yawn_model.predict(mouth_input, verbose=0)
                        yawn_label = "yawn" if yawn_pred[0][0] > 0.65 else "no_yawn"
                        yawn_status = yawn_label

                        if yawn_label == "yawn":
                            yawn_flag = True

                        my_adjusted = my + h // 2
                        cv2.rectangle(frame, (x + mx, y + my_adjusted), (x + mx + mw, y + my_adjusted + mh), (0, 0, 255), 2)
                        cv2.putText(frame, yawn_label, (x + mx, y + my_adjusted - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    except Exception as e:
                        print("Mouth error:", e)
            else:
                yawn_status = "yawn"

    else:
        if now - last_face_time > FACE_MISSING_THRESHOLD:
            face_missing_flag = True
            eye_status = "No Face"
            yawn_status = "Unknown"

    # Yawn logic
    if yawn_status == "yawn":
        yawn_frame_counter += 1
        if yawn_frame_counter >= 5:
            yawn_yawn_counter += 1
            yawn_frame_counter = 0

    if now - yawn_timer_start >= 10:
        if yawn_yawn_counter >= 5:
            play_beep()
        yawn_yawn_counter = 0
        yawn_timer_start = now

    # Flags
    if no_eye_detected_time is not None and now - no_eye_detected_time > NO_EYE_THRESHOLD:
        no_eye_flag = True

    if eye_closed_start_time is not None:
        closed_duration = now - eye_closed_start_time
        if closed_duration >= EYE_CLOSED_ALARM_DURATION:
            eye_closed_flag = True
    else:
        closed_duration = 0

    if eye_status != previous_eye_status:
        if previous_eye_status is not None:
            blink_count += 1
        previous_eye_status = eye_status

    if blink_count <=5 and elapsed_time >=10:
        winsound.Beep(1000, 500)
        blink_timer_start = current_time
        blink_count = 0
    elif elapsed_time > 10:
        blink_timer_start = current_time
        blink_count = 0

    
    # Trigger regular alarms
    if any([eye_closed_flag, yawn_yawn_counter>=5, face_missing_flag, no_eye_flag, blink_rate_flag]) and now - last_beep_time > 5:
        play_beep()
        last_beep_time = now
    if len(green_signal) >= 150 and time.time() - last_hr_update > 5:
        signal = np.array(green_signal)
        signal = detrend(signal)  # Remove trend
        fft_result = np.abs(fft(signal))
        freqs = np.fft.fftfreq(len(signal), d=1/30)  # Assuming ~30 fps

    # Only consider valid human pulse range (0.8–3 Hz or ~48–180 BPM)
        mask = (freqs > 0.8) & (freqs < 3)
        if np.any(mask):
            peak_freq = freqs[mask][np.argmax(fft_result[mask])]
            heart_rate = str(int(peak_freq * 60))+"BPM"  # Hz to BPM
            last_hr_update = time.time()


  

    # Display info
    cv2.putText(frame, f"Eye_status: {eye_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (148, 0, 211), 2)
    cv2.putText(frame, f"Yawn_status: {yawn_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Eye_Closed Time: {int(closed_duration)}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 100), 2)
    cv2.putText(frame, f"Blinks: {blink_count}/30s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (101, 67, 33), 2)
    cv2.putText(frame, f"Yawns (10s): {yawn_yawn_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (155, 135, 12), 2)
    cv2.putText(frame, f"Heart Rate: {heart_rate}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


    
    # Alerts
    if face_missing_flag:
        cv2.putText(frame, "ALERT: Face Missing > 5s!", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)
  
    if eye_closed_flag and eye_status=="closed":
        cv2.putText(frame, "ALERT: Eyes Closed > 5s!", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)
    if yawn_yawn_counter>=5:
        cv2.putText(frame, "ALERT: Yawning Detected!", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)
    if blink_count <= 5 and elapsed_time >=10:
        cv2.putText(frame, "ALERT: Low Blink Rate!", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 255), 2)
   
       # Show the frame
    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

