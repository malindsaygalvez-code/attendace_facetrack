import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset'

# ✅ CHECK kung may dataset folder
if not os.path.exists(path):
    print("❌ ERROR: dataset folder not found!")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    # Dito natin kukunin ang listahan ng folders (Arjie, Brithny, etc.)
    # os.walk ang gagamitin para pasukin pati sub-folders
    faceSamples = []
    ids = []
    
    # Gagawa tayo ng dictionary para i-map ang Name sa ID (kailangan ng number sa training)
    # Halimbawa: Arjie = 0, Brithny = 1, etc.
    names_map = {}
    current_id = 0

    # I-loop ang bawat folder sa loob ng dataset
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                # Kunin ang pangalan ng folder bilang label
                label_name = os.path.basename(root)
                
                # I-assign ang ID sa pangalan kung wala pa
                if label_name not in names_map:
                    names_map[label_name] = current_id
                    current_id += 1
                
                imagePath = os.path.join(root, file)
                
                try:
                    # I-convert sa grayscale
                    PIL_img = Image.open(imagePath).convert('L')
                    img_numpy = np.array(PIL_img, 'uint8')
                    
                    # Kunin ang ID para sa folder na ito
                    id = names_map[label_name]

                    # Detect faces
                    faces = detector.detectMultiScale(img_numpy)

                    for (x, y, w, h) in faces:
                        faceSamples.append(img_numpy[y:y+h, x:x+w])
                        ids.append(id)
                        
                    print(f"✅ Trained image: {imagePath} (ID: {id})")
                except Exception as e:
                    print(f"⚠️ Error processing {imagePath}: {e}")
                    continue

    # I-print ang mapping para alam mo kung anong ID ang bawat tao
    print("\n👥 Name Mapping:", names_map)
    return faceSamples, ids

print("🔄 Training faces...")

faces, ids = getImagesAndLabels(path)

# ✅ CHECK kung may na-detect na faces
if len(faces) == 0:
    print("❌ ERROR: No faces found in dataset! Siguraduhing may .jpg/.png sa loob ng folders.")
    exit()

recognizer.train(faces, np.array(ids))

# ✅ SAVE sa file
recognizer.save('trainer.yml')

print(f"\n✅ Training Complete! {len(np.unique(ids))} individual(s) found.")
print("📁 trainer.yml created successfully!")