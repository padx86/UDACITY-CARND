# Dependencies
import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os as os
import pickle as pickle
from moviepy.editor import VideoFileClip

class cameraCalibrator():
    """cameraCalibrator calibrates a camera with a given directory of camera calibration images using a chessboard for calibration"""
    def __init__(self, calibrationDirectory='./dummy', cbInnerCorners=(9,6)):
        self._path = calibrationDirectory
        self._innerCorners = cbInnerCorners
        self._cornerPath = '/corners/'
        self._testPath = '/cal_test/'
        self._savedFile = './CameraCoeffs.pkl'
        self.distortionCoeffs = None
        self.calibrationMatrix = None

    def checkPath(self):
        """checks if calibration directory exists"""
        if self._path[-1] is '/':
            self._path = self._path[:-1]                                #remove last slash if present

        if not os.path.isdir(self._path):
            raise Exception('Directory ' + self._path + ' not found')   #check if directory exists

    def prepCalibration(self):
        self._parentPath = self._path[:(self._path.rfind('/'))]         #get parent directoy path

        if os.path.exists(self._parentPath + self._cornerPath):         #check if directory exists containing images with detected corners
            for img in os.listdir(self._parentPath + self._cornerPath): 
                os.remove(self._parentPath + self._cornerPath + img)    #clear images from directory
        else:
            os.mkdir(self._parentPath + self._cornerPath)               #create diretory
        if os.path.exists(self._parentPath + self._testPath):
            for img in os.listdir(self._parentPath + self._testPath):
                    os.remove(self._parentPath + self._testPath + img)  #check if directory exists containing images to test calibration
        else:
            os.mkdir(self._parentPath + self._testPath)                 #create diretory

    def calibrate(self, saveCalibration=True, saveImages=True):
        self.checkPath()

        if saveImages:
            self.prepCalibration()
        
        imgpoints = []
        objpoints = []

        img_not_used_4_cal = []

        objpoint = np.zeros((self._innerCorners[0]*self._innerCorners[1],3),np.float32)
        objpoint[:,:2] = np.mgrid[0:self._innerCorners[0],0:self._innerCorners[1]].T.reshape(-1,2)
        
        cal_files = os.listdir(self._path)
        
        for cal_img in cal_files:
            img = mpimg.imread(self._path + '/' + cal_img)
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray_img, self._innerCorners, cv2.CALIB_CB_ADAPTIVE_THRESH)
            if ret:
                imgpoints.append(corners)
                objpoints.append(objpoint)
                if saveImages:
                    imgWithCorners = cv2.drawChessboardCorners(img, self._innerCorners, corners, ret)
                    mpimg.imsave(self._parentPath + self._cornerPath + cal_img, imgWithCorners)
            else:
                print('Board was not recognized on image:', cal_img)
                img_not_used_4_cal.append(cal_img)

        print('Images with detected corners saved to:', (self._parentPath + self._cornerPath))
        ret, cal_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1], None, None)
    
        if len(img_not_used_4_cal) & saveImages:
            for test_img in img_not_used_4_cal:
                img = mpimg.imread(self._path + '/' + test_img)
                dst = cv2.undistort(img, cal_matrix, dist_coeffs, None, cal_matrix)
                f, arr = plt.subplots(1,2,figsize=(10,4))
                arr[0].imshow(img)
                arr[0].set_title('Original image')
                arr[1].imshow(dst)
                arr[1].set_title('Undistorted image')
                plt.savefig(self._parentPath + self._testPath + test_img)
        print('undistorted Images were saved to:', (self._parentPath + self._testPath))
        self.calibrationMatrix = cal_matrix
        self.distortionCoeffs = dist_coeffs
        if saveCalibration:
            self.saveCalCoeffs(self._savedFile)
    
    def saveCalCoeffs(self, filePath):
        with open(filePath,'wb') as f:
            pickle.dump([self.calibrationMatrix, self.distortionCoeffs], f)
        f.close

    def loadCalCoeffs(self, filePath):
        with open(filePath,'rb') as f:
            pic = pickle.load(f)
            self.calibrationMatrix = pic[0]
            self.distortionCoeffs = pic[1]
        f.close

    def getCalibrationCoeffs(self, forceCalibration=False):
        if (os.path.exists(self._savedFile)) & (forceCalibration==False):
            if self.calibrationMatrix is None:
                self.loadCalCoeffs(self._savedFile)
        else:
            self.calibrate()
        return self.calibrationMatrix, self.distortionCoeffs

class laneline():
    """Container holding paramters for the lane detection history"""
    def __init__(self,left_fit,right_fit,left_fit_world,right_fit_world,center_fit,center_fit_world,curverad,average_lane_width):
        self.left_fit = left_fit
        self.right_fit = right_fit
        self.left_fit_world = left_fit_world
        self.right_fit_world = right_fit_world
        self.center_fit = center_fit
        self.center_fit_world = center_fit_world
        self.curverad = curverad
        self.average_lane_width = average_lane_width
        self.isDetected = False

class laneFinder():
    """laneFinder Findes lane lines on images and videos"""
    def __init__(self,calibrationMatrix,distortionCoeffs):
        #General
        self.frameNumber      = 1  #FrameCounter for Video Frames
        
        # Image undistortion
        self.calibrationMatrix = calibrationMatrix
        self.distortionCoeffs = distortionCoeffs
        
        #Thresholding
        self.sobel_kernel = 5
        self.thres_abs_sobel  = (20,100)
        self.thres_mag_sobel  = (20,100)
        self.thres_dir_sobel  = (0.8,1.5)
        self.thres_sat        = (120,255)
        
        #Warping
        self.wrpSrcPoints     = np.float32([[577,460],[203,720],[1127,720],[704,460]])
        self.wrpDestPoints    = np.float32([[320,0],[320,720],[960,720],[960,0]])
        
        #Pixel identification
        self.firstframe       = True
        self.isDetected       = False
        self.firstFrameMargin = 50
        self.firstFrameMinPix = 50
        self.nWindows         = 9

        # Quality
        self.lane_width_range = [2.6,4.2] #Allowed Lane width
        self.curv_range       = []
        self.lanelines        = []
        self.errorFrames      = 0

        #Transform
        self.ym_per_pix = 30 / 720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700 # meters per pixel in x dimension

    def findOnVideo(self, videoFilePath, saveToFile = None):
        """Finds lanes on a given video and writes an augmented Video with a specified file path"""
        if not os.path.exists(videoFilePath):                           #Check if file exists
            raise Exception('File ' + videoFilePath  + ' not found')
        self.isVideo = True                                             #Set isVideo flag
        clip1 = VideoFileClip(videoFilePath)                            #Read Video File
        white_clip = clip1.fl_image(self.find)                          #Find Lane for every image using find method handle
        if saveToFile is not None:
            white_clip.write_videofile(saveToFile, audio=False)         #Write video file to file path

    def findOnImage(self, imageFilePath, saveToFile = None):
        """Findes lanes on a given image and writes an augmented image with a specified file path"""
        if not os.path.exists(imageFilePath):                           #Check if file exists
            raise Exception('File ' + imageFilePath  + ' not found')    
        self.isVideo = False                                            #Set isVideo flag
        self.firstframe = True                                          #Set firstframe to use the function identifying first frames
        self.find(mpimg.imread(imageFilePath))                          #Find lane lines on image
        if saveToFile is not None:
            mpimg.imsave(saveToFile)                                    #Write image file to file path
        
    def find(self,img):
        """Controls all methods to necessary to find lane lines on a given image"""
        self.orig_image = img                                                                                                           #Store original image to class
        self.undist_image = cv2.undistort(self.orig_image, self.calibrationMatrix, self.distortionCoeffs, None, self.calibrationMatrix) #undistort original image
        self.gray_img = cv2.cvtColor(self.undist_image,cv2.COLOR_RGB2GRAY)                                                              #convert image to grayscale
        self.hls_img = cv2.cvtColor(self.undist_image,cv2.COLOR_RGB2HLS)                                                                #convert image to hls color space
        self.calcAllThreshold()                                                 #Calculate all desired thresholds
        self.combineThreshold()                                                 #Combine threshold results
        self.warped_img = self.warp_perspective(img = self.cmb)                 #Warp binary image
        if self.firstframe == True:                                             
            self.detectRelevantPixlesFirstFrame()                               #Process initial frame or frame if no lanes were detected for a certain number of previous framse
        else:
            self.detectRelevantPixles()                                         #Process frame if the starting point of the image is known
        self.polyfitLanes()                                                     #Polyfit lane lines
        self.convertToWorld()                                                   #Convert fit to 'birdview' coordinate system 
        self.polyLaneCenter()                                                   #Calculate Lane centers
        self.calcCurvature()                                                    #Calculate curvature
        self.calcAvgLaneWidth()                                                 #Calculate Average Lane width
        self.fitQuality()                                                       #Do sanity check
        self.applyDetectionToUndistImage()                                      #Augment undistorted image with additional information [lane lines,curvature,...]
        self.frameNumber += 1                                                   #increase framen umber
        return self.augmented_undist_img                                        #return augmented image
    
    def calcAllThreshold(self):
        """Processes all thresholding images and stores them to class instance"""
        self.astx = self.absSobelThresh(orient='x')                            
        self.asty = self.absSobelThresh(orient='y')
        self.mst  = self.magSobelThresh(img = self.hls_img[:,:,2])
        self.dst  = self.dirSobelThresh()
        self.sat  = self.satThresh() 

    def absSobelThresh(self, img = None, orient = 'x', sobel_kernel = None, thresh = None):
        """Calculates absolute sobel for a specific 2D direction (x,y)"""
        if img is None:                                             
            img = self.gray_img                     #Take from instance if not given
        if sobel_kernel is None:
            sobel_kernel = self.sobel_kernel        #Take from instance if not given
        if thresh is None:
            thresh = self.thres_abs_sobel           #Take from instance if not given
        
        if orient is 'x':
            sob = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=sobel_kernel) 
        elif orient is 'y':
            sob = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=sobel_kernel)
        else:
            raise Exception('orient must be x or y')

        abs_sob = np.absolute(sob)                                              #Get absolute sobel
        scaled_sob = np.uint8(255*abs_sob/np.max(abs_sob))                      #scale sobel
        grad_binary = np.zeros_like(scaled_sob)             
        grad_binary[(scaled_sob >= thresh[0]) & (scaled_sob <= thresh[1])] = 1  #Get values within threshold
        return grad_binary

    def magSobelThresh(self, img = None, sobel_kernel = None, thresh = None):
        """Calculates the magnitude of x and y sobel"""
        if img is None:
            img = self.gray_img                     #Take from instance if not given
        if sobel_kernel is None:
            sobel_kernel = self.sobel_kernel        #Take from instance if not given
        if thresh is None:
            thresh = self.thres_mag_sobel           #Take from instance if not given

        sobx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        soby = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        mag = np.sqrt(sobx**2 + soby**2)                                        #Calculate magnitude
        scaled_mag = np.uint8(255*mag/np.max(mag))                              #Scale magnitude
        mag_binary = np.zeros_like(scaled_mag)                                  
        mag_binary[(scaled_mag >= thresh[0]) & (scaled_mag <= thresh[1])] = 1   #Get values within threshold
        return mag_binary

    def dirSobelThresh(self, img = None, sobel_kernel = None, thresh = None):
        """Calculates the direction of x and y sobel"""
        if img is None:
            img = self.gray_img                     #Take from instance if not given
        if sobel_kernel is None:
            sobel_kernel = self.sobel_kernel        #Take from instance if not given
        if thresh is None:
            thresh = self.thres_dir_sobel           #Take from instance if not given
        
        sobx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel) #Get sobel in X
        soby = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel) #Get sobel in y
        dir = np.arctan2(np.absolute(soby),np.absolute(sobx))       #Cet direction
        dir_binary = np.zeros_like(dir)
        dir_binary[(dir >= thresh[0]) & (dir <= thresh[1])] = 1     #Get values within threshhold
        return dir_binary

    def satThresh(self, img = None, thresh = None):
        """Calculates a binary image of a channel(default: saturation cahnnel)"""
        if img is None:
            img = self.hls_img          #Take from instance if not given
        if thresh is None:
            thresh = self.thres_sat     #Take from instance if not given
        
        s_channel = img[:,:,2]                                              #Get the saturation channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1   #Get values within threshold
        return s_binary

    def combineThreshold(self):
        """Combines all Thresholds calculated previously"""
        self.cmb = np.uint8((((self.astx==1) & (self.asty==1)) | ((self.mst==1) & (self.dst==1)) & (self.sat==1)))   #Combine Thresholds to output a combined binary image
        return self.cmb

    def warp_perspective(self, img = None , srcPoints = None, destPoints = None, switch_src_dest = False):
        """Warps the Perspective of an image from srcPoints to destPoints in 2D, optionally switching direction of source and destination"""
        if img is None:
            img = self.undist_image         #Take from instance if not given       
        if srcPoints is None:
            srcPoints = self.wrpSrcPoints   #Take from instance if not given
        if destPoints is None:
            destPoints = self.wrpDestPoints #Take from instance if not given

        if len(img.shape)==3:
            img_size = (img.shape[0:-1][1],img.shape[0:-1][0])                      #Get correct image size for multi channel images
        else:
            img_size = (img.shape[1],img.shape[0])                                  #Get correct image size for single channel images

        if switch_src_dest:
            M = cv2.getPerspectiveTransform(destPoints,srcPoints)                   #Get inverse warp matrix
        else:
            M = cv2.getPerspectiveTransform(srcPoints, destPoints)                  #Get warp matrix
        
        warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)  #Warp image
        
        return warped_img

    def detectRelevantPixlesFirstFrame(self, img = None, margin = None, minpix = None, nwindows = None):
        """Detects relevant lane line pixles on the binary warped image"""
        if img is None:
            img = self.warped_img           #Take from instance if not given
        if margin is None:
            margin = self.firstFrameMargin  #Take from instance if not given
        if minpix is None:
            minpix = self.firstFrameMinPix  #Take from instance if not given
        if nwindows is None:
            nwindows = self.nWindows        #Take from instance if not given
        if len(img.shape) > 2:
            raise Exception('Image must be binary')

        self.histogram = np.sum(img[img.shape[0]//2:,:], axis=0)                    #create histogram
    
        midpoint = np.int(self.histogram.shape[0]/2)                                #center of histogram
        leftx_base = np.argmax(self.histogram[:midpoint])                           #get max of left side of histogram
        rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint               #get max of right side of histogram
        window_height = np.int(img.shape[0]/nwindows)                               #calc height of window
        
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        self.out_img = np.uint8(np.dstack((img, img, img))*255)         #Initialize output image

        for window in range(nwindows):
            win_y_low = img.shape[0] - (window+1)*window_height         #lower window boundary
            win_y_high = img.shape[0] - window*window_height            #upper window boundary
            win_xleft_low = leftx_current - margin                      #left lane line box left boundary
            win_xleft_high = leftx_current + margin                     #left lane line box right boundary
            win_xright_low = rightx_current - margin                    #right lane line box left boundary
            win_xright_high = rightx_current + margin                   #right lane line box right boundary
            
            cv2.rectangle(self.out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)      #draw left box
            cv2.rectangle(self.out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)    #draw right box

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]      #Get nonzero pixles in left box
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]   #Get nonzero pixles in right box
            
            left_lane_inds.append(good_left_inds)           #Append these indices to the left lane list
            right_lane_inds.append(good_right_inds)         #Append these indices to the right lane list
            
            if len(good_left_inds) > minpix:                                
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))       #relocate next left box if eneugh nonzero pixels were detected
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))     #relocate next right box if eneugh nonzero pixels were detected

        self.left_lane_inds = np.concatenate(left_lane_inds)
        self.right_lane_inds = np.concatenate(right_lane_inds)

        self.leftx = nonzerox[self.left_lane_inds]                              #Coordinates of relevant left lane pixels in x direction
        self.lefty = nonzeroy[self.left_lane_inds]                              #Coordinates of relevant left lane pixels in y direction
        self.rightx = nonzerox[self.right_lane_inds]                            #Coordinates of relevant right lane pixels in x direction
        self.righty = nonzeroy[self.right_lane_inds]                            #Coordinates of relevant right lane pixels in y direction
        self.out_img[self.lefty, self.leftx] = [255, 0, 0]                      #draw pixels to output image
        self.out_img[self.righty, self.rightx] = [0, 0, 255]                    #draw pixels to output image

        self.firstframe = False                                                 #Set firstframe flag to false
    
    def detectRelevantPixles(self,img = None, margin = None):
        """Detects relevant pixels of right and left lane lines if the starting point is set(firstframe flag is False)"""
        if img is None:
            img = self.warped_img                   #Take from instance if not given
        if margin is None:
            margin = self.firstFrameMargin          #Take from instance if not given
        if len(img.shape) > 2:
            raise Exception('Image must be binary')

        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] - margin)) & 
        (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))                          #Get left lane inds

        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] - margin)) & 
        (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))                       #Get right lane inds

        # Again, extract left and right line pixel positions
        self.leftx = nonzerox[left_lane_inds]       #Coordinates of relevant left lane pixels in x direction
        self.lefty = nonzeroy[left_lane_inds]       #Coordinates of relevant left lane pixels in y direction
        self.rightx = nonzerox[right_lane_inds]     #Coordinates of relevant right lane pixels in x direction
        self.righty = nonzeroy[right_lane_inds]     #Coordinates of relevant right lane pixels in y direction

    def polyfitLanes(self,order=2):
        """Polyfit LaneLines with given order"""
        if len(self.lefty)>0:
            self.left_fit = np.polyfit(self.lefty, self.leftx, order)
        else:
            self.left_fit = [0.0, 0.0, 0.0]
        if len(self.righty)>0:
            self.right_fit = np.polyfit(self.righty, self.rightx, order)
        else:
            self.right_fit = [0.0, 0.0, 0.0]
    
    def polyLaneCenter(self):
        """Calculates the lane center im image and world coordinates weighted by the number of relevant pixels detected"""
        self.center_fit = np.zeros_like(self.right_fit)
        self.center_fit[2]=self.left_fit[2]+(self.right_fit[2]-self.left_fit[2])/2                                                      #Offset of the center
        self.center_fit[1]=(self.left_fit[1]*len(self.lefty)+self.right_fit[1]*len(self.righty))/(len(self.lefty)+len(self.righty))      #Linear coeff
        self.center_fit[0]=(self.left_fit[0]*len(self.lefty)+self.right_fit[0]*len(self.righty))/(len(self.lefty)+len(self.righty))      #Square coeff
        self.center_fit_world = np.zeros_like(self.center_fit)
        self.center_fit_world[2]=self.left_fit_world[2]+(self.right_fit_world[2]-self.left_fit_world[2])/2                              #Offset of the center
        self.center_fit_world[1]=(self.left_fit_world[1]*len(self.lefty)+self.right_fit_world[1]*len(self.righty))/(len(self.lefty)+len(self.righty))      #Linear coeff
        self.center_fit_world[0]=(self.left_fit_world[0]*len(self.lefty)+self.right_fit_world[0]*len(self.righty))/(len(self.lefty)+len(self.righty))      #Square coeff

    def convertToWorld(self,order=2):
        """Converts the fit to world coordinates('birdview') using constants self.ym_per_pix & self.xm_per_pix"""
        
        ploty = np.linspace(0, self.warped_img.shape[0]-1, self.warped_img.shape[0])            #Create coordinates in y
        leftx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]           #create coordinates in x left lane
        rightx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]       #create coordinates in x right lane

        self.left_fit_world = np.polyfit(ploty*self.ym_per_pix, leftx*self.xm_per_pix, order)   #refit polynomial to world left lane
        self.right_fit_world = np.polyfit(ploty*self.ym_per_pix, rightx*self.xm_per_pix, order) #refit polynomial to world left lane

    def calcCurvature(self):
        """Calculates indvidual curveture radius for left and right lane and combines them using the number of relevant pixels detected as weights"""
        y_eval = max([max(self.lefty),max(self.righty)])
        self.left_curverad = ((1 + (2*self.left_fit_world[0]*y_eval*self.ym_per_pix + self.left_fit_world[1])**2)**1.5) / np.absolute(2*self.left_fit_world[0])         #calculate left lane curvateure
        self.right_curverad = ((1 + (2*self.right_fit_world[0]*y_eval*self.ym_per_pix + self.right_fit_world[1])**2)**1.5) / np.absolute(2*self.right_fit_world[0])     #calculate right lane curvature 
        self.curverad = (self.left_curverad*len(self.lefty)+self.right_curverad*len(self.righty))/(len(self.lefty)+len(self.righty))                                    #Combine curvetures
    
    def calcAvgLaneWidth(self,nSamples=5):
        """Calculates the average lane width by using a given number of evenly distributed samples"""
        ploty = np.linspace(0, self.warped_img.shape[0]-1, nSamples)*self.ym_per_pix                                #Create points in y
        left_fitx = self.left_fit_world[0]*ploty**2 + self.left_fit_world[1]*ploty + self.left_fit_world[2]         #Calc points in x for left lane
        right_fitx = self.right_fit_world[0]*ploty**2 + self.right_fit_world[1]*ploty + self.right_fit_world[2]     #Calc points in x for right lane
        self.average_lane_width = sum(abs(left_fitx-right_fitx)/nSamples)                                           #use mean difference of the points for average_lane_width

    def fitQuality(self):
        """Decides wether the fit is valid and uses a previous fit if not"""
        ll = laneline(left_fit = self.left_fit,right_fit = self.right_fit, left_fit_world = self.left_fit_world,
        right_fit_world = self.right_fit_world, center_fit = self.center_fit, center_fit_world = self.center_fit_world,
        curverad = self.curverad,average_lane_width = self.average_lane_width)                                        #Create lane line with information

        if (self.laneWidthsOkay()):     #lane width valid?
            ll.isValid = True
            if self.isDetected == False:    #Check if at least 10 valid Frames were detected in a row to to turn on the detection if it was thrown off or startet
                if len(self.lanelines)>=10:
                    for i in range(9):
                        self.isDetected = True
                        if self.lanelines[i].isValid == False:
                            self.isDetected = False
                            break

        else:                                   #lane width invalid
            ll.isValid = False                  #set invalid flag to lane line
            mpimg.imsave('./errorFrames/FrameNr'+ str(self.frameNumber) + '.jpg', self.orig_image)                          #Save errorframe for analysis
            self.errorFrames +=1                                                                                            #increment errorframe counter
            if (ll.average_lane_width < self.lane_width_range[0]) | (ll.average_lane_width > self.lane_width_range[1]):     #Reset if average lanewidth gets out of bounds
                self.firstframe = True
            if self.isDetected:                                                                                             #if the system is on
                if len(self.lanelines)>=10:                                                                                 #double check but must be fullfilled to turn on system
                    mpimg.imsave('img/' + str(self.frameNumber) + '.jpg',self.orig_image)
                    for i in range(9):                                                                                      #Use most recent valid frame data, but store frame as invalid
                        self.isDetected = False
                        self.firstframe = True
                        if self.lanelines[i].isValid:
                            self.isDetected = True
                            self.firstframe = False
                            self.left_fit = self.lanelines[i].left_fit                                                      #recall valid data
                            self.right_fit = self.lanelines[i].right_fit                                                    #recall valid data
                            self.left_fit_world = self.lanelines[i].left_fit_world                                          #recall valid data
                            self.right_fit_world = self.lanelines[i].right_fit_world                                        #recall valid data
                            self.center_fit = self.lanelines[i].center_fit                                                  #recall valid data
                            self.center_fit_world = self.lanelines[i].center_fit_world                                      #recall valid data
                            self.curverad = self.lanelines[i].curverad                                                      #recall valid data
                            self.average_lane_width = self.lanelines[i].average_lane_width                                  #recall valid data
                            break
        
        self.lanelines.insert(0,ll)
        
        if len(self.lanelines) > 15:            #Remove lanelines older than 15 Frames
            self.lanelines = self.lanelines[0:14]

    def laneWidthsOkay(self, testSamples=5):
        """Check if lane widths are in a given range at a given number of sample points evenly spread"""
        
        ploty = np.linspace(0, self.warped_img.shape[0]-1, testSamples)*self.ym_per_pix
        left_fitx = self.left_fit_world[0]*ploty**2 + self.left_fit_world[1]*ploty + self.left_fit_world[2]
        right_fitx = self.right_fit_world[0]*ploty**2 + self.right_fit_world[1]*ploty + self.right_fit_world[2]
        return all(((right_fitx-left_fitx)>self.lane_width_range[0]) & ((right_fitx-left_fitx)<self.lane_width_range[1]))

    def applyDetectionToUndistImage(self):
        """Augments image with additional data such as lane lines, lane width, curvature radius"""
        if (self.isDetected) | (self.isVideo==False):
            overlay = np.zeros_like(self.undist_image)
            if len(self.lefty)>0:
                overlay[self.lefty, self.leftx] = [255, 0, 0]       #Overlay with valid pixles if detected
                minly = int(min(self.lefty))                        #get valid pixel with highest distance to the car
            else:
                minly = 1
            if len(self.righty)>0:
                overlay[self.righty, self.rightx] = [255, 0, 0]     #Overlay with valid pixles if detected
                minry = min(self.righty)                            #get valid pixel with highest distance to the car
            else:
                minry = 1
                
            plotyl = np.linspace(minly, overlay.shape[0]-1, overlay.shape[0]-minly)                                                 #create y coordinates for left lane line
            plotyr = np.linspace(minry, overlay.shape[0]-1, overlay.shape[0]-minry)                                                 #create y coordinates for right lane line
            plotyc = np.linspace(int((minly+minry)/2),overlay.shape[0]-1,overlay.shape[0]-int((minly+minry)/2))                     #create y coordinates for lane center
            left_fitx = self.left_fit[0]*plotyl**2 + self.left_fit[1]*plotyl + self.left_fit[2]                                     #create x coordinates for left lane line
            right_fitx = self.right_fit[0]*plotyr**2 + self.right_fit[1]*plotyr + self.right_fit[2]                                 #create x coordinates for right lane line
            center_fitx = self.center_fit[0]*plotyc**2 + self.center_fit[1]*plotyc +self.center_fit[2]                              #create x coordinates for lane center
            cv2.polylines(overlay, np.int32([np.column_stack((left_fitx,plotyl))]),isClosed=False,color=[0,255,0],thickness=8)      #draw left lane line
            cv2.polylines(overlay, np.int32([np.column_stack((right_fitx,plotyr))]),isClosed=False,color=[0,255,0],thickness=8)     #draw right lane line
            cv2.polylines(overlay, np.int32([np.column_stack((center_fitx,plotyc))]),isClosed=False,color=[255,255,0],thickness=8)  #draw lane center
            warped_overlay = self.warp_perspective(img=overlay,switch_src_dest=True)                                                #warp overlay back to fit undistorted original
            ret,mask = cv2.threshold(cv2.cvtColor(warped_overlay,cv2.COLOR_RGB2GRAY),10,255,cv2.THRESH_BINARY_INV)                  #create mask
            masked_undist = cv2.bitwise_and(self.undist_image,self.undist_image,mask = mask)                                        #mask undistorted original
            self.augmented_undist_img = cv2.add(masked_undist,warped_overlay)                                                       #overlay undistorted original
            dist2center = (self.augmented_undist_img.shape[1]/2-center_fitx[-1])*self.xm_per_pix                                    #calculate distance 2 center
            cv2.putText(self.augmented_undist_img,('LaneWidth [m]: ' + str(self.average_lane_width)), (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)   #add avg lane width
            cv2.putText(self.augmented_undist_img,('Curve Radius [m]: ' + str(self.curverad)), (30,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)          #add curv radius
            cv2.putText(self.augmented_undist_img,('Distance from center [m]: ' + str(dist2center)),(30,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)      #add distance from center
        else:
            self.augmented_undist_img = self.undist_image

        if self.isVideo:
            cv2.putText(self.augmented_undist_img,('ErrorFrames: ' + str(self.errorFrames)),(30,160),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)              #add number of errorframes if is video

    def plotDetection(self,savepath=None):
        """Plots a 3x3 Matrix to visualize single step results, usefull for parameter fitting and error frame analysis"""
        f, plotArr = plt.subplots(3,3,figsize=(15,15))
        plotArr[0,0].imshow(self.undist_image)
        plotArr[0,0].set_title('Undistorted Image with warping area')
        x_coor,y_coor = [],[]
        for line in self.wrpSrcPoints:
            x_coor.append(line[0])
            y_coor.append(line[1])
        x_coor.append(x_coor[0])
        y_coor.append(y_coor[0])
        plotArr[0,0].plot(x_coor,y_coor)
        plotArr[1,0].imshow(self.warp_perspective(self.undist_image))
        plotArr[1,0].set_title('Warped image')
        x_coor,y_coor = [],[]
        for line in self.wrpDestPoints:
            x_coor.append(line[0])
            y_coor.append(line[1])
        x_coor.append(x_coor[0])
        y_coor.append(y_coor[0])
        plotArr[1,0].plot(x_coor,y_coor)
        
        plotArr[2,0].imshow(self.astx)
        plotArr[2,0].set_title('Absolute Sobel in X-Direction')
        plotArr[0,1].imshow(self.asty)
        plotArr[0,1].set_title('Absolute Sobel in Y-Direction')
        plotArr[1,1].imshow(self.mst)
        plotArr[1,1].set_title('Absolute Sobel Magnitude')
        plotArr[2,1].imshow(self.dst)
        plotArr[2,1].set_title('Absolute Sobel Direction')
        plotArr[0,2].imshow(self.sat)
        plotArr[0,2].set_title('Saturation Threshold')
        
        ploty = np.linspace(0, self.warped_img.shape[0]-1, self.warped_img.shape[0])
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        plotArr[1,2].imshow(self.out_img)
        plotArr[1,2].set_title('Warped Threshold image with relevant pixles')
        plotArr[1,2].plot(left_fitx, ploty, color='yellow')
        plotArr[1,2].plot(right_fitx, ploty, color='yellow')
        
        plotArr[2,2].imshow(self.augmented_undist_img)
        plotArr[2,2].set_title('Warped Threshold image with relevant pixles')
        if savepath is not None:
            plt.savefig(savepath)
        plt.show()

### Calibrate Camera ###
c = cameraCalibrator('CarND-Advanced-Lane-Lines/camera_cal')
calibrationMatrix, distortionCoeffs = c.getCalibrationCoeffs(forceCalibration=True)

### Create class and find lanes on Video ###
l = laneFinder(calibrationMatrix,distortionCoeffs)
l.findOnVideo('CarND-Advanced-Lane-Lines/project_video.mp4','project_video_augmented.mp4')

### Take a look at all test images ###
example_dir = 'test_images'
### Take a look at all Error Frames ###
#example_dir = 'errorFrames'

### Uncomment for looking at images ###
#example_list = os.listdir(example_dir)
#for imgFile in example_list:
#    l.findOnImage(example_dir + '/' + imgFile)
#    l.plotDetection('./img/' + imgFile)
    