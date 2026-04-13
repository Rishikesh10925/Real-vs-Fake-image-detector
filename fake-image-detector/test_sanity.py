
import os, cv2, numpy as np
import tensorflow as tf
from utils import CLASS_NAMES, IMG_SIZE

model = tf.keras.models.load_model('fake_real_detector.h5')
real_dir = '../real_and_fake_face/training_real'

files = os.listdir(real_dir)[:30]
count_fake = 0
count_real = 0

for f in files:
    img = cv2.imread(os.path.join(real_dir, f))
    if img is None: continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    prob = model.predict(img, verbose=0)[0][0]
    if prob >= 0.5:
        count_real += 1
    else:
        count_fake += 1

print(f'Tested {len(files)} real images. Predicted Real: {count_real}, Predicted Fake: {count_fake}')

