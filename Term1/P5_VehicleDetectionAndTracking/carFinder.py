## Dependencies
# General
import numpy as np
import time
import os
import glob
import pickle
import sklearn
#Sklearn
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
#CV2 and Matplotlib and scipy and moviepy
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

class car():
    """Helper to store data of passed detections"""
    def __init__(self,min_x,max_x,min_y,max_y):
       self.max_x = max_x                           #Maximum boundary of car in x
       self.min_x = min_x                           #Minimum boundary of car in x
       self.min_y = min_y                           #Minimum boundary of car in y
       self.max_y = max_y                           #Maximum boundary of car in y
       self.width = max_x-min_x                     #Width of car
       self.height = max_y-min_y                    #Height of car
       self.centroid_x = (max_x-min_x)/2 + min_x    #centroid of car in x
       self.centroid_y = (max_y-min_y)/2 + min_y    #centroid of car in y
       self.parent = None                           #parent object, if detected
       self.parentcount = 0                         #Number of parents

class carFinder():
    def __init__(self):
        self.spaces_possible = ('RGB','HSV','LUV','HLS','YUV','YCrCb','GRAY')  #Possible color spaces
        # default values
        # --> these values can be tuned to get better results
        self.color_spaces = ['HSV']*1             #Color spaces used to define what colorspace the image shall have for extractHist
        self.color_channels = ['HSV']*1           #Corresponding Channels to define what channels of self.colorspaces shall be used by extractHist
        self.hist_bins = 16                         #hist_bins used by extractHist
        self.hist_range = (0,256)                   #hist_range used by extractHist
        self.spatial_size = (8,8)                   #Spatial size used by extractSapcial
        self.hog_orient = 7                         #Orientations used by extractHog
        self.hog_pix_per_cell = 8                   #Pixels per cell used by extractHog
        self.hog_cell_per_block = 4                 #Cells per block used by extractHog
        self.hog_spaces = ['YCrCb']*1           #Color spaces used to define what colorspace the image shall have for extractHog
        self.hog_channels = ['YCrCb']*1           #Corresponding Channels to define what channels of self.colorspaces shall be used by extractHog
        self.sw_region = None                       #Sliding window region: Tuple=(x_low,x_high,y_low,y_high)
        self.sw_shape = (128,128)                   #Sliding window size of box
        self.sw_overlap = (0.5,0.5)                 #Sliding window overlap of box
        self.use_hist = False                       #Shall histograms be used for features
        self.use_hog = True                         #Shall hog be used for features
        self.use_spatial = False                    #Shall bined color features be used
        #default values
        #--> not to be changed, allowed read write just for experimentation
        self.classifier = None                      #Classifier storage
        self.scaler = None                          #Scaler storage
        self.classifier_file_path = 'clf_dump.pkl'  #Path classifier/scaler is saved to
        self.test_accuracy = 0                      #Test accuracy of training
        self.feature_count = 0                      #Number of features used 
        self.test_sample_count = 0                  #Number of samples tested on
        self.train_sample_count = 0                 #Number of samples trained on
        self.car_files_path = '../../data/car/'     #Training data for car images
        self.nocar_files_path = '../../data/nocar/' #Training data for non-car images
        self.isVideo = False                        #is a Video been fed or an image
        self.previous_frame_cars = None             #Frame containing previous object detections
        
        
    ## Shared section
    def transformColorSpace(self, img, desiredColor = 'RGB', showImg = False):
        """Transform Color Spaces from RGB Images"""
        if desiredColor not in self.spaces_possible:
            raise Exception('Colorspace unkown')

        if desiredColor == 'GRAY':
            img_cspace = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)       #Transform image to Gray color space 
        elif desiredColor == 'HSV':
            img_cspace = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)       #Transform image to [Hue,Saturation,Value] color space 
        elif desiredColor == 'LUV':
            img_cspace = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)       #Transform image to [L,U,V] color space
        elif desiredColor == 'HLS':                                 
            img_cspace = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)       #Transform image to [Hue,Light,Saturation] color space
        elif desiredColor == 'YUV':
            img_cspace = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)       #Transform image to [Y,U,V] color space
        elif desiredColor == 'YCrCb':
            img_cspace = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)     #Transform image to [Y,Cr,Cb] color space -> [luma, red difference chroma, blue difference chroma] very similar to YUV
        else: 
            img_cspace = np.copy(img)
        if showImg:                                                 #Show color space in pyplot figure
            if desiredColor=='GRAY':
                print('Gray images will not be shown')
            else:
                if desiredColor =='YCrCb':
                    c1,c2,c3 = 'Y','Cr','Cb'
                else:
                    c1,c2,c3 = desiredColor[0],desiredColor[1],desiredColor[2]
                f,arr = plt.subplots(2,2,figsize=(12,8))
                arr[0,0].imshow(img_cspace[:,:,0],cmap='gray')
                arr[0,0].set_title(c1)
                arr[1,0].imshow(img_cspace[:,:,1],cmap='gray')
                arr[1,0].set_title(c2)
                arr[0,1].imshow(img_cspace[:,:,2],cmap='gray')
                arr[0,1].set_title(c3)
                arr[1,1].imshow(img_cspace)
                arr[1,1].set_title(desiredColor)
                #plt.savefig('../../color_spaces/' + desiredColor + '.jpg')
                plt.show()

        return img_cspace

    def extractSpatial(self, img, spatial_size = None):
        """Returns spatial features"""
        #Get default values
        if spatial_size == None:
            spatial_size = self.spatial_size

        img = cv2.resize(img,spatial_size)  #resize image
        features = np.ravel(img)            #flatten resized image
        return features                     

    def extractHist(self, img, hist_bins = None, hist_range = None):
        """Returns histogramm features of a one or three channel image"""
        #Get default values if required
        if hist_bins == None:
            hist_bins = self.hist_bins
        if hist_range == None:
            hist_range = self.hist_range

        if len(img.shape)==3:
            c1,c2,c3 = np.histogram(img[:,:,0], bins=hist_bins, range=hist_range), np.histogram(img[:,:,1], bins=hist_bins, range=hist_range), np.histogram(img[:,:,2], bins=hist_bins, range=hist_range)
            histogram_features = np.concatenate((c1[0], c2[0], c3[0]))                  #Create histograram features of three channel image
        else:
            histogram_features = np.histogram(img, bins = hist_bins, range = hist_range)   
            histogram_features = np.concatenate((histogram_features))                   #Create histogram of one channel image
        return histogram_features

    def extractHog(self, img, orient = None, pix_per_cell = None, cell_per_block = None, visualize = False, feature_vect = True):
        """Returns Hog features on one given Channel"""
        #Get default values if required
        if orient == None:
            orient = self.hog_orient
        if pix_per_cell == None:
            pix_per_cell = self.hog_pix_per_cell
        if cell_per_block == None:
            cell_per_block = self.hog_cell_per_block
        
        if len(img.shape) != 2:
            raise Exception('Wrong image dimensions')

        #Calculate hog features
        if visualize == True:
            hog_features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=True, feature_vector=False)
            return hog_features, hog_image
        else:      
            hog_features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=False, feature_vector=feature_vect)
            return hog_features
    
    def showHogFeatures(self, img, orient = None, pix_per_cell= None, cell_per_block = None, color_space='RGB'):
        """Shows a plot of each channel in hog representation"""
        img_hog = self.transformColorSpace(img,desiredColor = color_space) #Trandform color space
        hc1,hi1 = self.extractHog(img_hog[:,:,0], orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block, visualize = True) #Create Hog for first channel
        hc2,hi2 = self.extractHog(img_hog[:,:,1], orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block, visualize = True) #Create Hog for second channel
        hc3,hi3 = self.extractHog(img_hog[:,:,2], orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block, visualize = True) #Create hog for third channel
        
        #Plot results and add titles
        f, arr = plt.subplots(2,2,figsize=(12,8))
        if color_space =='YCrCb':
            c1,c2,c3 = 'Y','Cr','Cb'
        else:
            c1,c2,c3 = color_space[0],color_space[1],color_space[2]
        f.suptitle('HOG with orient='+str(orient)+',pixpercell='+str(pix_per_cell)+',cellperblock='+str(cell_per_block))
        arr[0,0].imshow(img_hog)
        arr[0,0].set_title('Image used for Hog: ' + color_space)
        arr[0,1].imshow(hi1)
        arr[0,1].set_title('Hog on: ' + c1)
        arr[1,0].imshow(hi2)
        arr[1,0].set_title('Hog on: ' + c2)
        arr[1,1].imshow(hi3)
        arr[1,1].set_title('Hog on: ' + c3)
        #plt.savefig('../../hog_images/' + color_space + '.jpg')
        plt.show()

    def extractFeatures(self, img, color_spaces = None , color_channels = None, spatial_size = None , hist_bins = None, hist_range = None, hog_orient = None, hog_pix_per_cell = None, hog_cell_per_block = None, hog_spaces = None, hog_channels = None, use_hog = None, use_hist = None, use_spatial = None):
        """Extracts features from given images, uses default settings in init if None, By defining 
        spaces and corresponding channels features from single channels in different spaces can be extracted
        Example: hog_spaces=('RGB','HSV') ; hog_channels=('RB','V') will return the hog represantation of the R & B Channel in RGB Space and the HOG of the V Channel in HSV space"""
        #Get default values if required
        if color_spaces == None:
            color_spaces = self.color_spaces
        if color_channels == None:
            color_channels = self.color_channels
        if hog_spaces == None:
            hog_spaces = self.hog_spaces
        if hog_channels == None:
            hog_channels = self.hog_channels
        if use_hist == None:
            use_hist = self.use_hist
        if use_hog == None:
            use_hog = self.use_hog
        if use_spatial == None:
            use_spatial = self.use_spatial

        #Check if valid parameter lengths were given
        if len(color_spaces) != len(color_channels):
            raise Exception("Color Channels and Spaces must be the same size")
        if len(hog_spaces) != len(hog_channels):
            raise Exception("Hog Channels and Spaces must be the same size")

        feature_list = []

        #Get spatial features
        if use_spatial:
            spatial_features = self.extractSpatial(img,spatial_size = spatial_size) #Get Spatial features
            feature_list.append(spatial_features)

        #Get hist features
        if use_hist:
            for c_space,c_chan in zip(color_spaces,color_channels):
                c_img = self.transformColorSpace(img, desiredColor = c_space)
                if (c_space==c_chan) | (c_space=='GRAY'):
                    hist_features = self.extractHist(c_img, hist_bins = hist_bins, hist_range = hist_range)
                    feature_list.append(hist_features)
                elif cspace == 'YCrCb':
                    if 'Y' in h_chan:
                        hist_features = self.extractHist(c_img[:,:,0], hist_bins = hist_bins, hist_range = hist_range)
                        feature_list.append(hist_features)
                    if 'Cr' in h_chan:
                        hist_features = self.extractHist(c_img[:,:,1], hist_bins = hist_bins, hist_range = hist_range)
                        feature_list.append(hist_features)
                    if 'Cb' in h_chan:
                        hist_features = self.extractHist(c_img[:,:,2], hist_bins = hist_bins, hist_range = hist_range)
                        feature_list.append(hist_features)
                else:
                    for chan in c_chan:
                        hist_features = self.extractHist(c_img[:,:,c_space.index(chan)], hist_bins = hist_bins, hist_range = hist_range)
                        feature_list.append(hist_features)
        
        #Get hog features
        if use_hog:
            for h_space,h_chan in zip(hog_spaces,hog_channels):
                c_img = self.transformColorSpace(img, desiredColor = h_space)
                if h_space == 'GRAY':
                    hog_features = self.extractHog(c_img, orient = hog_orient, pix_per_cell = hog_pix_per_cell,cell_per_block = hog_cell_per_block)
                    feature_list.append(hog_features)
                elif h_space == 'YCrCb':
                    if 'Y' in h_chan:
                        hog_features = self.extractHog(c_img[:,:,0], orient = hog_orient, pix_per_cell = hog_pix_per_cell,cell_per_block = hog_cell_per_block)
                        feature_list.append(hog_features)
                    if 'Cr' in h_chan:
                        hog_features = self.extractHog(c_img[:,:,1], orient = hog_orient, pix_per_cell = hog_pix_per_cell,cell_per_block = hog_cell_per_block)
                        feature_list.append(hog_features)
                    if 'Cb' in h_chan:
                        hog_features = self.extractHog(c_img[:,:,2], orient = hog_orient, pix_per_cell = hog_pix_per_cell,cell_per_block = hog_cell_per_block)
                        feature_list.append(hog_features)
                else:
                    for chan in h_chan:
                        hog_features = self.extractHog(c_img[:,:,h_space.index(chan)], orient = hog_orient, pix_per_cell = hog_pix_per_cell,cell_per_block = hog_cell_per_block)
                        feature_list.append(hog_features)
            return np.concatenate(feature_list)

    ## Training section
    def loadImageSet(self, car_files_path = None, nocar_files_path = None, maxImages = None):
        """Loads *.png Images for training/testing a classifier"""
        #load default values if required
        if car_files_path == None:
            car_files_path = self.car_files_path
        if nocar_files_path == None:
            nocar_files_path = self.nocar_files_path

        #Get file generators
        car_files = glob.iglob(car_files_path + '/**/*.png', recursive=True)
        nocar_files = glob.iglob(nocar_files_path + '/**/*.png',recursive=True)
        
        #Define empty lists
        car_images =[]
        nocar_images = []

        for idx,cimg in enumerate(car_files):
            img = mpimg.imread(cimg)    #Read image of car
            img = np.uint8(img*256)     #Convert to uint8 image
            car_images.append(img)      #append image to list
            if maxImages != None:
                if idx >= maxImages:
                    break
        print('car images loaded')

        for idx,ncimg in enumerate(nocar_files):
            img = mpimg.imread(ncimg)   #Read image of nocar
            img = np.uint8(img*256)     #Convert to uint8 image
            nocar_images.append(img)    #append image to list
            if maxImages != None:
                if idx >= maxImages:
                    break
        print('non-car images loaded')

        #Make them the same length
        if len(car_images)>len(nocar_images):
            car_images = car_images[0:len(nocar_images)]
        else:
            nocar_images = nocar_images[0:len(car_images)]
        
        #Shuffle and return
        return sklearn.utils.shuffle(car_images, nocar_images)

    def createClassifier(self, loadclassfier = True, classifier_type = 'LinearSVC'):
        """Creates or loads a classifier"""
        if (os.path.exists(self.classifier_file_path)) & loadclassfier:     #Loads trained classifier if desired and existant
            with open(self.classifier_file_path,'rb') as f:
                pic = pickle.load(f)
                self.classifier = pic[0]
                self.scaler = pic[1]
                self.test_accuracy = pic[2]
                self.feature_count = pic[3]
                self.train_sample_count = pic[4]
                self.test_sample_count = pic[5]
                print('trained classifier loaded and readdy for use')
            f.close()
        else:                                                               #Else create new classifier
            if classifier_type == 'LinearSVC':
                self.classifier = LinearSVC()
                print('Untrained classifier created: You need to train it before use')
            else:
                raise Exception('Only LinearSVC is implemented at this time')
    
    def trainClassifier(self, car_files_path = None, nocar_files_path = None, maxImages = None, test_size = 0.2, useOptimalClassifier = True, use_hist = None, use_spatial = None, use_hog = None):
        """Loads training images, extracts features, trains and tests the classifier defined in the createClassifier Method"""

        #Load training/testing images
        print("Begining training operation")
        t0 = time.time()
        car_images, nocar_images = self.loadImageSet(car_files_path = car_files_path, nocar_files_path = nocar_files_path, maxImages = maxImages)
        t_load = time.time() - t0
        t0 = time.time()

        #Extract features
        car_features = []
        nocar_features = []
        for c_img in car_images:
            car_features.append(self.extractFeatures(c_img, use_hist = use_hist, use_hog = use_hog, use_spatial = use_spatial))
        print('Finished extracting car features')
        for nc_img in nocar_images:
            nocar_features.append(self.extractFeatures(nc_img, use_hist = use_hist, use_hog = use_hog, use_spatial = use_spatial))
        print('Finished extracting non-car features')
        combined_features = np.vstack((car_features,nocar_features)).astype(np.float64)
        self.scaler = StandardScaler().fit(combined_features)
        scaled_features = self.scaler.transform(combined_features)
        print('Finished scaling features')
        t_feat = time.time()-t0
        labels = np.hstack((np.ones(len(car_features)), np.zeros(len(nocar_features))))
        print('Finished creating labels')
        
        #Create Training and testing data
        X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=test_size, random_state=np.random.randint(0, 100))
        self.feature_count = len(X_train[0])
        self.train_sample_count = len(X_train)
        self.test_sample_count = len(X_test)
        
        #Train classifier
        print('Beginning to train')
        t0 = time.time()
        self.classifier.fit(X_train, y_train)
        t_train = time.time()-t0
        print('Finished training')

        #Test classifier
        t0 = time.time()
        self.test_accuracy = round(self.classifier.score(X_test, y_test), 4)
        t_test = time.time()-t0
        print('finished testing')

        #Save classifier
        print('Saving classifier to:',self.classifier_file_path)
        with open(self.classifier_file_path,'wb') as f:
            pickle.dump([self.classifier, self.scaler ,self.test_accuracy, self.feature_count,self.train_sample_count,self.test_sample_count], f)
        f.close()

        #Print out summary
        print('###############################################')
        print('SUMMARY')
        print('###############################################')
        print('Loading',len(car_images),'car images and',len(nocar_images),'no car images took',t_load,'seconds')
        print('Feature extraction on',len(car_images),'car images and',len(nocar_images),'no car images took',t_feat,'seconds')
        print('Feature extraction resulted in a feature vector length of:',self.feature_count)
        print('Testing on', self.test_sample_count,'took',t_test,'seconds and resulted in a test accuracy of:', self.test_accuracy)

    ## Prediction Section
    def slidingWindows(self,img,region = None, window_shape = None, window_overlap = None):
        """Sliding Windows Method equivalent to the one used in UDACITY Project introduction"""
        # Get default values if required
        if region == None:
            if self.sw_region == None:
                region = (0,img.shape[1],0,img.shape[0])
            else:
                region = self.sw_region
        if window_shape == None:
            window_shape = self.sw_shape
        if window_overlap == None:
            window_overlap = self.sw_overlap
            
        
        xspan = region[1] - region[0]   #Compute span of ROI in x    
        yspan = region[3] - region[2]   #Compute span of ROI in x
        
        nx_pix_per_step = np.int(window_shape[0]*(1 - window_overlap[0]))   #Compute the number of pixels per step in x
        ny_pix_per_step = np.int(window_shape[1]*(1 - window_overlap[1]))   #Compute the number of pixels per step in y
        # Compute the number of windows in x/y
        nx_buffer = np.int(window_shape[0]*(window_overlap[0]))
        ny_buffer = np.int(window_shape[1]*(window_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
        # Initialize a list to append window positions to
        window_list = []
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + region[0]
                endx = startx + window_shape[0]
                starty = ys*ny_pix_per_step + region[2]
                endy = starty + window_shape[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows required for feature searching
        return window_list

    def drawBoxes(self, img, bboxes, color=(0, 255, 255), thick=6):
        """Draws a list of boxes onto an image"""
        imcopy = np.copy(img)
        for bbox in bboxes:
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        return imcopy
    
    def searchClassic(self, img, region = None, window_shape = None, window_overlap = None):
        """Extracts features and predicts if there is a car on the patch revieved from the sliding windows method or not"""
        if region is None:
            region = (0,img.shape[1],350,img.shape[0])

        windows = self.slidingWindows(img, region = region , window_shape = window_shape, window_overlap= window_overlap)
        onwindows = []
        for window in windows:
            features = self.extractFeatures(cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64)), use_hist = use_hist, use_hog = use_hog, use_spatial = use_spatial)
            scaled_features = self.scaler.transform(np.array(features).reshape(1, -1))
            prediction = self.classifier.predict(scaled_features)
            if prediction:
                onwindows.append(window)
        return onwindows

    def searchFullHog(self, img, scale = 1, region = None, hog_spaces = None, hog_channels = None, orient = None, cell_per_block = None, pix_per_cell = None, color_spaces = None, color_channels = None, use_hist = None, use_hog = None, use_spatial = None):
        """search Full Hog was the search method used for the submission, it was introduced in the UDACITY Project introduction and modified for personal use
        it creates HOG representation of the complete ROI before searching for swifter HOG feature extraction"""
        #Get default if required
        if orient == None:
            orient = self.hog_orient
        if cell_per_block == None:
            cell_per_block = self.hog_cell_per_block
        if pix_per_cell == None:
            pix_per_cell = self.hog_pix_per_cell
        if hog_spaces == None:
            hog_spaces = self.hog_spaces
        if hog_channels == None:
            hog_channels = self.hog_channels
        if color_spaces == None:
            color_spaces = self.color_spaces
        if color_channels == None:
            color_channels = self.color_channels
        if use_hist == None:
            use_hist = self.use_hist
        if use_hog == None:
            use_hog = self.use_hog
        if use_spatial == None:
            use_spatial = self.use_spatial

        img_search = img[region[2]:region[3],region[0]:region[1],:]     #Set region of interest
        if scale != 1:                                                  #Scale image if necassary
            img_search = cv2.resize(img_search, (np.int(img_search.shape[1]/scale), np.int(img_search.shape[0]/scale)))
        hog_list = []

        #Convert all required images and get HOG representation
        for h_space,h_chan in zip(hog_spaces,hog_channels):
            c_img = self.transformColorSpace(img_search,desiredColor=h_space)
            if h_space == 'GRAY':
                hog_features = self.extractHog(c_img, orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block, feature_vect = False)
                hog_list.append(hog_features)
            elif h_space == 'YCrCb':
                if 'Y' in h_chan:
                    hog_features = self.extractHog(c_img[:,:,0], orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block, feature_vect = False)
                    hog_list.append(hog_features)
                if 'Cr' in h_chan:
                    hog_features = self.extractHog(c_img[:,:,1], orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block, feature_vect = False)
                    hog_list.append(hog_features)
                if 'Cb' in h_chan:
                    hog_features = self.extractHog(c_img[:,:,2], orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block, feature_vect = False)
                    hog_list.append(hog_features)
            else:
                for chan in h_chan:
                    hog_features = self.extractHog(c_img[:,:,h_space.index(chan)], orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block, feature_vect = False)
                    hog_list.append(hog_features)

        # Define blocks and steps as above
        nxblocks = (c_img.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (c_img.shape[0] // pix_per_cell) - cell_per_block + 1 
        nfeat_per_block = orient*cell_per_block**2
    
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        bboxes = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feature_list = []
                for hog in hog_list:
                    hog_feature_list.append(hog[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel())
                
                hog_features = np.hstack((hog_feature_list))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(img_search[ytop:ytop+window, xleft:xleft+window], (64,64))
          
                #Extract color features
                spatial_features = []
                if use_spatial:
                    spatial_features = self.extractSpatial(subimg, spatial_size = spatial_size)

                hist_feature_list = []
                hist_features = []
                #Extract histogram features
                if use_hist:
                    for c_space,c_chan in zip(color_spaces,color_channels):
                        c_img = self.transformColorSpace(subimg, desiredColor = c_space)
                        if (c_space==c_chan) | (c_space=='GRAY'):
                            hist_feat = self.extractHist(c_img, hist_bins = hist_bins, hist_range = hist_range)
                            hist_feature_list.append(hist_feat)
                        elif cspace == 'YCrCb':
                            if 'Y' in h_chan:
                                hist_features = self.extractHist(c_img[:,:,0], hist_bins = hist_bins, hist_range = hist_range)
                                feature_list.append(hist_features)
                            if 'Cr' in h_chan:
                                hist_features = self.extractHist(c_img[:,:,1], hist_bins = hist_bins, hist_range = hist_range)
                                feature_list.append(hist_features)
                            if 'Cb' in h_chan:
                                hist_features = self.extractHist(c_img[:,:,2], hist_bins = hist_bins, hist_range = hist_range)
                                feature_list.append(hist_features)
                        else:
                            for chan in c_chan:
                                hist_feat = self.extractHist(c_img[:,:,c_space.index(chan)], hist_bins = hist_bins, hist_range = hist_range)
                                hist_feature_list.append(hist_feat)
                    hist_features = np.hstack((hist_feature_list))

                stacked = np.hstack((spatial_features, hist_features, hog_features)).reshape(1,-1)  #stack all features
                scaled_features = self.scaler.transform(stacked)                                    #Normalize all features
                prediction = self.classifier.predict(scaled_features)                               #Predict if car or not
                
                #If a car was predicted 
                if prediction:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    bboxes.append(((xbox_left, ytop_draw+region[2]),(xbox_left+win_draw,ytop_draw+win_draw+region[2]))) #Append box to bounding boxes of detected images
        #return bounding boxes
        return bboxes

    def heatmapAndLabel(self, img, bboxes, threshold = 2):

        heatmap = np.zeros_like(img[:,:,0])
        for box in bboxes:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        heatmap[heatmap<threshold]=0
        labels = label(heatmap)
        return labels 

    def drawLabeledBboxesVid(self,img, labels):
        # Iterate through all detected cars and save to frame_cars_detected
        frame_cars_detected = []
        for car_number in range(1, labels[1]+1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            c = car(np.min(nonzerox), np.max(nonzerox), np.min(nonzeroy), np.max(nonzeroy))
            c.ID = car_number
            frame_cars_detected.append(c)
        
        if self.previous_frame_cars is not None:
            for c in frame_cars_detected:
                for c2 in self.previous_frame_cars:
                    if (abs(c2.centroid_x-c.centroid_x)<70.0) & (abs(c2.centroid_y-c.centroid_y)<70.0):
                        c.parentcount = c2.parentcount+1
                        c.parent = c2
                        if c.parentcount > 15:               #Discard older parents
                            c.parent.parent.parent = None
                            c.parentcount = 15
                        break
        carsfound = 0
        for c in frame_cars_detected:
            if c.parentcount >= 15:
                c2 = c.parent
                c3 = c2.parent
                cv2.rectangle(img, (np.int(0.7*c.min_x+0.2*c2.min_x+0.1*c3.min_x),np.int(0.7*c.min_y+0.2*c2.min_y+0.1*c3.min_y)), (np.int(0.7*c.max_x+0.2*c2.max_x+0.1*c3.max_x),np.int(0.7*c.max_y+0.2*c2.max_y+0.1*c3.max_y)), (0,0,255), 6)
                carsfound +=1
        cv2.putText(img,'Cars_identified: ' + str(carsfound),(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)  #Add Number of cars found to image
        self.previous_frame_cars = frame_cars_detected
        return img

    def drawLabeledBboxes(self,img,labels):
        """Draws bounding boxes on videos and smoothens them"""
         # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        cv2.putText(img,'Cars_identified:' + str(labels[1]),(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
        return img

    def findCarsOnImage(self,img_file_path):
        """Finds cars on images"""
        if not os.path.exists(img_file_path):
            raise Exception('File not found')
        self.isVideo = False
        return self.find(mpimg.imread(img_file_path))

    def findCarsOnVideo(self,video_file_path,saveToFile = None):
        """Finds cars on videos"""
        if not os.path.exists(video_file_path):                         #Check if file exists
            raise Exception('File not found')
        self.isVideo = True                                             #Set isVideo Flag
        clip1 = VideoFileClip(video_file_path)                          #Read Video File
        white_clip = clip1.fl_image(self.find)                          #Find Lane for every image using find method handle
        if saveToFile is not None:
            white_clip.write_videofile(saveToFile, audio=False)         #Write video file to file path

    def find(self,img):
        """Runs feature extraction, classification and displays found objects on a single image"""
        #Get Bounding Boxes for Objects
        cars_1 = self.searchFullHog(img,scale=0.8,region = (0,img.shape[1],350,500))
        cars_2 = self.searchFullHog(img,scale=1.0,region = (0,img.shape[1],350,500))
        cars_3 = self.searchFullHog(img,scale=1.2,region = (0,img.shape[1],400,500))
        cars_4 = self.searchFullHog(img,scale=1.4,region = (0,img.shape[1],300,550))
        cars_5 = self.searchFullHog(img,scale=1.6,region = (0,img.shape[1],350,550)) 

        cars = cars_1+cars_2+cars_3+cars_4+cars_5                                           #Combine found boxes
        labels =self.heatmapAndLabel(img,cars)                                              #create heatmaps and label objects
        if self.isVideo:
            augmented_img = self.drawLabeledBboxesVid(img,labels)                           #Use this method to draw boxes on images if is a video
        else:
            augmented_img = self.drawLabeledBboxes(img,labels)                              #Use this method to draw boxes on images if is a single image
            augmented_img = self.drawBoxes(augmented_img,cars)
        return augmented_img

#### Main
cF = carFinder()

#Configuration
cF.use_hog = True 
cF.use_hist = False 
cF.use_spatial = False
#### Use to train new classifier ###
#cF.createClassifier(loadclassfier=False)
#cF.trainClassifier(maxImages=10000)
#cF.findCarsOnVideo('../../CarND-Vehicle-Detection/project_video.mp4',saveToFile='../../project_video.mp4')

#### Use to load ready trained classifier
#cF.createClassifier()
## Video
#cF.findCarsOnVideo('../../CarND-Vehicle-Detection/project_video.mp4',saveToFile='../../project_video.mp4')
## Image
#plt.imshow(cF.findCarsOnImage('../../CarND-Vehicle-Detection/test_images/test1.jpg'))
#plt.show()