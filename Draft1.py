from emnist import extract_test_samples

# EMNIST letters chunk dataset
# Has 145600 images
print("imported EMNIST libraries")

# x images, y labels
x, y = extract_test_samples('letters')

# convert pixels values from 0-255 to 0-1
# so that easier for NN to process
x = x / 255

# first 60000 images as training set, next 10000 images as test set
x_train, x_test = x[:18000], x[18000:20800]
# first 60000 labels as training set, next 10000 labels as test set
y_train, y_test = y[:18000], y[18000:20800]

# record number of samples in each dataset
# and number of pixels in each image
x_train = x_train.reshape(18000, 784)
x_test = x_test.reshape(2800, 784)

print("Extracted samples. Divided training and testing data sets.")

# images are 28x28 pixels
# each pixel is greyscale value between 0 and 255

#import matplotlib.pyplot as plt

