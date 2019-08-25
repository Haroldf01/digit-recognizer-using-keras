import matplotlib.pyplot as plt
from keras.datasets import mnist

(aTrainData, bTrainData), (aTestData, bTestData) = mnist.load_data()

# print(plt.imshow(aTrainData[3]))
print(plt.imshow(aTrainData[12]))
# Number 3
plt.show()

'''
img = img.reshape((28,28))
plt.imshow(img)
plt.title(classname)
plt.show()
'''
