import cv2
from tools import get_hog_features
from random import randint
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def displayDifferentHOG(veh_images, nonveh_images):
	rand_indexes = []
	for i in range(10):
		 rand_indexes.append(randint(0, len(veh_images)))



	for i in range(len(rand_indexes)):
		vehicle = mpimg.imread(veh_images[rand_indexes[i]])
		#non_vehicle = mpimg.imread(nonveh_images[rand_indexes[i]])
		gray = cv2.cvtColor(vehicle, cv2.COLOR_RGB2GRAY)

		# Call our function with vis=True to see an image output
		features_or9, hog_image_or9 = get_hog_features(gray, orient= 9, 
								pix_per_cell= 8, cell_per_block= 2, 
								vis=True, feature_vec=False)
		
		features_or11, hog_image_or11 = get_hog_features(gray, orient= 11, 
								pix_per_cell= 8, cell_per_block= 2, 
								vis=True, feature_vec=False)
		
		features_or9_pix16, hog_image_or9_pix16 = get_hog_features(gray, orient= 9, 
								pix_per_cell= 16, cell_per_block= 2, 
								vis=True, feature_vec=False)
		
		features_or11_pix16, hog_image_or11_pix16 = get_hog_features(gray, orient= 11, 
								pix_per_cell= 16, cell_per_block= 2, 
								vis=True, feature_vec=False)


		# Plot the examples
		f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(32,32))
		ax1.imshow(vehicle, cmap='gray')
		ax1.set_title('Example Car Image', fontsize=12)

		ax2.imshow(hog_image_or9, cmap='gray')
		ax2.set_title('HOG OR: 9, Pix_cell: 8', fontsize=12)

		ax3.imshow(hog_image_or11, cmap='gray')
		ax3.set_title('HOG OR: 11, Pix_cell: 8', fontsize=12)

		ax4.imshow(hog_image_or9_pix16, cmap='gray')
		ax4.set_title('HOG OR: 9, Pix_cell: 16', fontsize=12)

		ax5.imshow(hog_image_or11_pix16, cmap='gray')
		ax5.set_title('HOG OR: 11, Pix_cell: 16', fontsize=12)

	rand_indexes = []
	for i in range(10):
		 rand_indexes.append(randint(0, len(nonveh_images)))
			
	for i in range(len(rand_indexes)):
		nonvehicle = mpimg.imread(nonveh_images[rand_indexes[i]])
		#non_vehicle = mpimg.imread(nonveh_images[rand_indexes[i]])
		gray = cv2.cvtColor(nonvehicle, cv2.COLOR_RGB2GRAY)

		# Call our function with vis=True to see an image output
		features_or9, hog_image_or9 = get_hog_features(gray, orient= 9, 
								pix_per_cell= 8, cell_per_block= 2, 
								vis=True, feature_vec=False)
		
		features_or11, hog_image_or11 = get_hog_features(gray, orient= 11, 
								pix_per_cell= 8, cell_per_block= 2, 
								vis=True, feature_vec=False)
		
		features_or9_pix16, hog_image_or9_pix16 = get_hog_features(gray, orient= 9, 
								pix_per_cell= 16, cell_per_block= 2, 
								vis=True, feature_vec=False)
		
		features_or11_pix16, hog_image_or11_pix16 = get_hog_features(gray, orient= 11, 
								pix_per_cell= 16, cell_per_block= 2, 
								vis=True, feature_vec=False)


		# Plot the examples
		f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(32,32))
		ax1.imshow(nonvehicle, cmap='gray')
		ax1.set_title('Example Non Car Image', fontsize=12)

		ax2.imshow(hog_image_or9, cmap='gray')
		ax2.set_title('HOG OR: 9, Pix_cell: 8', fontsize=12)

		ax3.imshow(hog_image_or11, cmap='gray')
		ax3.set_title('HOG OR: 11, Pix_cell: 8', fontsize=12)

		ax4.imshow(hog_image_or9_pix16, cmap='gray')
		ax4.set_title('HOG OR: 9, Pix_cell: 16', fontsize=12)

		ax5.imshow(hog_image_or11_pix16, cmap='gray')
		ax5.set_title('HOG OR: 11, Pix_cell: 16', fontsize=12)