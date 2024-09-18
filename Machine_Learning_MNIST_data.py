# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:40:47 2024

@author: Morgen
"""
#import list
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

#Loads training and test data
(training_i, training_l), (test_i, test_l) = mnist.load_data()

#Sets 'digits' to desired outcome. Applies a filter on trainging labels to just include 0,1, and 9
digits = [0,1,9]
training_filter = np.isin(training_l, digits)

#Filters images and labels so only 0,1, and 9 remain
filtered_i = training_i[training_filter]
filtered_l = training_l[training_filter]

#Informative, used in next step
print(f"Shape of filtered_i (images): {filtered_i.shape}")

#checks filtered shape
if filtered_i.shape[1:] == (28, 28): 
    
    #Reshapes all images into 784 columns
    x = filtered_i.reshape(filtered_i.shape[0], 28*28).T  

    #sets values as a float
    x = x.astype('float32')
    
    #output for troubleshooting purposes
    print(f"Shape of X: {x.shape}")  
    print(f"Data type of X: {x.dtype}")  
    
#safety for trouble shooting purposes
else:
    print("The shape of the filtered images is not as expected.")
    
#Norms

# Verify that the labels are correctly filtered, done for troubleshooting purposes
unique_labels, label_counts = np.unique(filtered_l, return_counts=True)
print("Unique labels after filtering:", unique_labels)
print("Counts of each label:", label_counts)

#Normilization of pixel intensity from 0-1
filtered_i = filtered_i / 255.0


averages = {}

for digit in [0, 1, 9]:
    #filter images based on digit
    digit_images = filtered_i[filtered_l == digit]
    average_image = np.mean(digit_images, axis=0)  
    averages[digit] = average_image
    overall_average = np.mean(digit_images)
    
    #print statement for troubleshooting purposes
    print(f"Average pixel intensity calculated for digit {digit} is {overall_average}")
    
    #needed for calculating norms, it gets angry without it being flat
    flat_av_image = average_image.flatten()
    
    #calculate norms
    L1_norm = np.linalg.norm(flat_av_image,1)
    print(f"L1 norm for {digit} is {L1_norm}")
    L2_norm = np.linalg.norm(flat_av_image,2)
    print(f"L2 norm for {digit} is {L2_norm}")
    L3_norm = np.linalg.norm(flat_av_image,3)
    print(f"L3 norm for {digit} is {L3_norm}")

#I was curios what the training data looked like so i outputted an image. Kinda neat.
'''
image_index = 10

image = filtered_i[image_index].reshape(28, 28)

plt.imshow(image, cmap='gray')
plt.title(f"Label: {filtered_l[image_index]}")
plt.show()
'''

U, S, Vt = np.linalg.svd(x, full_matrices=False)  # decompose X into U, S, and V^T

#values plotted in decreasing order
plt.figure(figsize=(9, 6))
plt.plot(S, marker='o')
plt.title("Singular Values of S")
plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.yscale('log')  # Used logarithmic scale to better visualize the decay
plt.grid(True)
plt.show()

#K set
K_set = [1, 51, 101, 151, 201, 251, 301, 351, 401, 451, 501, 551, 601, 651, 701, 751, 771, 781, 784]

approx_errors = [] #creates an empty list

for K in K_set:
    Qk = U[:, :K] #selects first vector from K
    Yk = Qk @ Qk.T @ x #matrix multiplication
    #error = abs(x - Yk)  Tried this originally, got weird output, then tried forbian norm, much better outputs
    error = np.linalg.norm(x - Yk, ord = 'fro')
    approx_errors.append(error)
    
    print(f"K={K}, Approx error: {error}")

#Plot output
plt.figure(figsize=(9, 6))
plt.plot(K_set, approx_errors, marker='o')
plt.title("Approximation Error vs K")
plt.xlabel("K (Number of Singular Vectors)")
plt.ylabel("Approximation Error (Frobenius Norm)")
plt.grid(True)
plt.show()

m, n = x.shape #caculates shape of x

elements_x = m * n #calculates total numbers of elements in x

storage_Yk = [] #creates an empty list

#storage calculations
for K in K_set:
    storage_Uk = m * K
    
    storage_Sk = K #Sk only contains top values of K
    
    storage_Vtk = K * n
    
    total_storage_Yk = storage_Uk + storage_Sk + storage_Vtk
    storage_Yk.append(total_storage_Yk)
    #Point of diminishing returns
    if K == 201:
        print(f"Total elements stored for K = 201: {total_storage_Yk} elements")
    #after K = 601 k_error goes to effectivly 0
    if K == 601:
        print(f"Total elements stored for K = 601: {total_storage_Yk} elements")

#plotting
plt.figure(figsize=(9, 6))
plt.plot(K_set, storage_Yk, marker='o', label='Storage for Yk')
plt.axhline(y=elements_x, color='r', linestyle='--', label='Storage for full X')
plt.title("Storage Requirements for Yk vs K")
plt.xlabel("K (Number of Singular Vectors)")
plt.ylabel("Number of Elements Stored")
plt.legend()
plt.grid(True)
plt.show()

#print statement for total storage needed
print(f"Storage required for full matrix X: {elements_x} elements")
print(f"Storage required for low-rank approximation Y_{K}: {total_storage_Yk} elements")

Q2 = U[:, :2]  #first 2 left vectors
projected_data = Q2.T @ x  # projects x onto u....i hate matrix math so much

#filters projected data
projected_by_digit = {digit: projected_data[:, filtered_l == digit] for digit in [0, 1, 9]}

#conputes centroids...again
centroid2 = {digit: np.mean(projected_by_digit[digit], axis=1) for digit in [0, 1, 9]}

#plotting scatter points
colors = {0: 'red', 1: 'blue', 9: 'green'}
markers = {0: 'o', 1: 's', 9: 'd'}
plt.figure(figsize=(9, 6))


for digit in [0, 1, 9]:
    plt.scatter(projected_by_digit[digit][0, :], projected_by_digit[digit][1, :], c=colors[digit], 
                label=f'Digit {digit}', marker=markers[digit], alpha=0.6)

#plots the centroid
centroid_colors = {0: 'darkred', 1: 'darkblue', 9: 'darkgreen'}  
centroid_marker = '*'  
for digit in [0, 1, 9]:
    plt.scatter(centroid2[digit][0], centroid2[digit][1], c=centroid_colors[digit], 
                marker=centroid_marker, s=400, edgecolor='white', label=f'Centroid {digit}')

#plot output
plt.title('Projection of Digits on the First Two Singular Vectors')
plt.xlabel('First Singular Vector')
plt.ylabel('Second Singular Vector')
plt.legend()
plt.grid(True)
plt.show()

#console output for centoid
for digit in [0, 1, 9]:
    print(f"Centroid for digit {digit}: {centroid2[digit]}")


#Same random prechosen seed for "reliably" random
np.random.seed(40)
xn = x.copy()

#noise introduction
noise = np.random.rand(*xn.shape) < 0.05 # 5% probability
xn[noise] = np.random.choice([0, 255], size = noise.sum())

#SVD seperation
Un, Sn, Vtn = np.linalg.svd(xn, full_matrices = False)

#Creates empty list for error calcs
error_xYk = []
error_xnYk = []
#forbian norm
error_xxn = np.linalg.norm(x - xn, ord = 'fro')

for K in K_set:
    Qk = Un[:, :K]
    Yk = Qk @ (Qk.T @ xn)
    
    error_xYk.append(np.linalg.norm(x- Yk, ord='fro'))
    error_xnYk.append(np.linalg.norm(xn - Yk, ord='fro'))

plt.figure(figsize = (9, 6))
plt.plot(K_set, error_xYk, marker = 'o', label = 'Error of (x, Yk)')
plt.plot(K_set, error_xnYk, marker = '*', label = 'Error of (xn, Yk)')
plt.axhline(y= error_xxn, color = 'r', linestyle = '--', label = 'Error of (x, xn)', alpha=0.8)
plt.title("Errors (x, Yk) and (Xn, Yk) vs K")
plt.xlabel("K (Number of Singular Vectors)")
plt.ylabel("Frobenius Norm (Error)")
plt.legend()
plt.grid(True)
plt.show()







