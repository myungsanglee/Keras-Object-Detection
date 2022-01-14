import os

import numpy as np
import cv2
import albumentations as A


a = np.zeros((2, 7), dtype=np.float32)
a[0:1, ...] = [1, 1, 1, 1, 1, 1, 1]
a[1:, ...] = [2, 2, 2, 2, 2, 2, 2]
print(a)

b = np.take(a, np.where(a[..., 1] > 2)[0], axis=0)
print(b)
print(b.shape)
if b.shape[0] == 0:
    print(f"There is b")
    
    
c = []
for i in range(5):
    c.append(float(i))
print(c)