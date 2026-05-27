import cv2
import sqlite3
from datetime import datetime
import time
import streamlit as st

def start_scanner(placeholder, confidence_threshold):

    # 1. LOAD MODELS
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("trainer.yml")

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return False

    # =====================================================
    # CONFIGURATION BASED ON were.png FOLDERS
    # =====================================================
    names = [
        "None",             # ID 0
        "Hazel Mae",        # ID 1
        "Fredirick",        # ID 2
        "Roxan",            # ID 3
        "Kristina",         # ID 4
        "Meaann",           # ID 5
        "Arjie",            # ID 6
        "Hinayon",          # ID 7
        "Brithny",          # ID 8
        "Lindsay Galvez",   # ID 9
        "Jenelyn"           # ID 10
    ]

    all_students = names[1:] 

    detected_students = set()
    last_mark_time = 0
    
    # Counter para sa tagal ng pagpapakita ng check mark sa screen
    check_display_expiry = 0

    # --- INTERNAL FUNCTIONS ---

    def mark_attendance(name):
        nonlocal last_mark_time
        if time.time() - last_mark_time < 5: # 5 seconds cooldown
            return False
        last_mark_time = time.time()

        is_new = False
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, date))
        if cursor.fetchone() is None:
            cursor.execute(
                "INSERT INTO attendance (name, date, time, status) VALUES (?, ?, ?, ?)",
                (name, date, current_time, "Present")
            )
            conn.commit()
            
            # --- REAL-TIME ALERT INSERTION ---
            st.toast(f"✅ Na-record na si {name}!", icon='👤')
            # ----------------------------------
            
            print(f"✅ {name} marked PRESENT!")
            is_new = True # Bagong attendance naisulat sa DB
            
        conn.close()
        return is_new

    def mark_absent():
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        for student in all_students:
            if student not in detected_students:
                cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (student, date))
                if cursor.fetchone() is None:
                    cursor.execute(
                        "INSERT INTO attendance (name, date, time, status) VALUES (?, ?, ?, ?)",
                        (student, date, current_time, "Absent")
                    )
        conn.commit()
        conn.close()

    # --- MAIN CAMERA LOOP ---
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 640) 
    cap.set(4, 480) 

    if not cap.isOpened():
        st.error("❌ Cannot open camera")
        return False

    st.session_state.scanner_running = True

    while st.session_state.get("scanner_running", True):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) # Flip para natural na salamin ang galaw
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray) 

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100)
        )

        # Basahin ang kasalukuyang prediction trigger button galing UI
        do_predict = st.session_state.get("do_prediction", False)

        for (x, y, w, h) in faces:
            # Default presentation state habang nakatutok lang ang camera
            name = "Ready to Scan"
            color = (255, 255, 255) # White habang naghihintay
            
            if do_predict:
                # 1. Instant Yellow State para sa "Scanning..." feedback loop
                st.session_state.last_result = "Scanning..."
                st.session_state.last_color = (0, 255, 255) # Yellow sa BGR
                
                name = st.session_state.last_result
                color = st.session_state.last_color
                
                # 2. Actual LBPH Prediction Execution
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                match_perc = round(100 - confidence)

                if match_perc >= confidence_threshold:
                    if id < len(names):
                        name = names[id]
                        detected_students.add(name)
                        
                        if mark_attendance(name):
                            check_display_expiry = time.time() + 1.5
                            
                        st.session_state.last_result = f"{name} ({match_perc}%)"
                        st.session_state.last_color = (0, 255, 0) # Green
                    else:
                        st.session_state.last_result = "Unknown ID"
                        st.session_state.last_color = (0, 0, 255) # Red
                else:
                    st.session_state.last_result = "Unknown"
                    st.session_state.last_color = (0, 0, 255) # Red
            else:
                # Kung tapos na mag-predict, i-retain ang nakaraang resulta sa screen ng mukha
                name = st.session_state.get("last_result", "Ready")
                color = st.session_state.get("last_color", (255, 255, 255))

            # I-guhit ang bounding box at pangalan mismo sa paligid ng mukha
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(frame, name, (x, y-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # I-reset ang prediction trigger pagkatapos ng loop cycle sa nakitang mukha
        if do_predict:
            st.session_state.do_prediction = False

        # --- DRAWING LOGIC PARA SA MALAKING BERDENG CHECK MARK (Kapag Success) ---
        if time.time() < check_display_expiry:
            cv2.line(frame, (260, 260), (300, 300), (0, 255, 0), 15)
            cv2.line(frame, (300, 300), (390, 190), (0, 255, 0), 15)
            cv2.putText(frame, "SUCCESS", (240, 350), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholder.image(frame_rgb, channels="RGB")

        if not st.session_state.get("scanner_running", True) or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    # CLEANUP
    cap.release()
    cv2.destroyAllWindows()
    mark_absent() 
    placeholder.empty()
    return True