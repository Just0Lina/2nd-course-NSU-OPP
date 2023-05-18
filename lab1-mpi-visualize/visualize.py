import numpy as np
import matplotlib.pyplot as plt
import sys


N = 50

x = np.fromfile("vecX.bin", dtype=np.float32).reshape((N, N))

xMy = np.fromfile("vecMyX.bin", dtype=np.float32).reshape((N, N))

x2 = np.fromfile("vecMyXClear.bin", dtype=np.float32).reshape((N, N))



b = np.fromfile("vecB.bin", dtype=np.float32).reshape((N, N))

a = np.fromfile("matA.bin", dtype=np.float32).reshape((N*N, N*N))

np.set_printoptions(threshold=sys.maxsize)
file = open("sample.txt", "w+")

# Saving the array in a text file
content = str(a)
file.write(content)
file.close()

# print("VecA\n", a)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4)

ax2.imshow(x)
ax2.set_title('Vector X')

ax3.imshow(xMy)
ax3.set_title('Vector My X 5 iter')

ax4.imshow(-b)
ax4.set_title('Vector B')

ax1.imshow(x2)
ax1.set_title('Vector X 1000 iter')

plt.show()