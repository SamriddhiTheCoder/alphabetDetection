import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps
import cv2

X = np.load("image.npz")["arr_0"]
Y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(Y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

sample_per_class = 5
fig = plt.figure(figsize=(nclasses*2, (1+sample_per_class*2)))

idx_cls = 0
for cls in classes:
  idxs = np.flatnonzero(Y == cls)
  idxs = np.random.choice(idxs, sample_per_class, replace=False)
  i = 0
  for idx in idxs:
    plt_idx = i * nclasses + idx_cls + 1
    p = plt.subplot(sample_per_class, nclasses, plt_idx)
    p = sns.heatmap(np.reshape(X[idx], (22, 30)), cmap=plt.cm.gray, xticklabels=False, yticklabels=False, cbar=False)
    p = plt.axis('off')
    i += 1
  idx_cls += 1

x_train, x_test, y_train,  y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
x_train_scale = x_train / 255.0
x_test_scale = x_test / 255.0

clf = LogisticRegression(solver="saga", multi_class="multinomial")
clf.fit(x_train_scale, y_train)
y_pred = clf.predict(x_test_scale)

cap = cv2.VideoCapture(0)

while(True):
    try: 
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRTOGRAY())
        height, width = gray.shape
        up_left = (int(width/2 - 56), int(height/2 - 56))
        bottom_right = (int(width/2 + 56), int(height/2 + 56))

        roi = gray[up_left[1]: bottom_right[1], up_left[0]: bottom_right[0]]
        im_pil = Image.fromarray(roi) 

        image_bw = im_pil.convert('L') 
        image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)

        pixel_filter = 20 
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter) 
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255) 
        max_pixel = np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel

        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        test_pred = clf.predict(test_sample) 
        print("Predicted class is: ", test_pred)

        cv2.imshow("frame", gray)
        if cv2.waitkey(5):
            break
    except Exception as e: pass

cv2.destroyAllWindows()
cap.release()

