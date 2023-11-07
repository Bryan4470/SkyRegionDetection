
import cv2
import numpy as np
import os
import random
from matplotlib import pyplot as pt
import shutil
import time


# This function loads a list with specified number of images and their file names 
def LoadDataset(imageDir, size=None):

    imList = []

    # Get the list of image files
    imageFiles = [file for file in os.listdir(imageDir)]

    #obtain random x images for processing
    if size is not None:
        imageFiles = random.sample(imageFiles, size)
        
    for file in imageFiles:
        # Construct the full path to the image file
        filePath = os.path.join(imageDir, file)

        # Read in the image
        im = cv2.imread(filePath)
        
        
        if im is not None:
            imList.append((im, file))  # Append image and file name as a tuple    
            
    return imList #list of (image, file name)

#Obtain gradient magnitude of image
def CalculateGradientMagnitude(image):
    
    img	= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    meanIntensity = np.mean(img)  
    lowThreshold = meanIntensity * 0.55
    highThreshold = 255
    
    #detect edges in image
    edges	= cv2.Canny(img, lowThreshold  ,highThreshold)
    
    # Calculate gradient magnitude
    gradientX = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
    gradientY = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
    gradientMag = np.sqrt(gradientX ** 2 + gradientY ** 2)

    return gradientMag

#Create mask
def CreateMask(b, image):
    mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
    for xx, yy in enumerate(b):
        mask[yy:, xx] = 255
        
    # invert mask so sky pixels assigned with 255, region of interest
    mask = cv2.bitwise_not(mask)

    return mask
    
#Calculate sky broder position function
def CalculateBorder(gradientImg, t): # pass in gradient image & parameter t
    sky = np.full(gradientImg.shape[1], gradientImg.shape[0]) #1D NumPy array of sky

    for x in range(gradientImg.shape[1]):
        borderPos = np.argmax(gradientImg[:, x] > t)

        if borderPos > 0:
            sky[x] = borderPos

    return sky

#Calculate energy function
def EnergyFunction(bTmp, image):
    #binary mask based on estimated border values
    skyMask = CreateMask(bTmp, image)
    
    #determine ground and sky region in the mask 
    groundRegion = np.ma.array(image, mask=cv2.cvtColor(cv2.bitwise_not(skyMask), cv2.COLOR_GRAY2BGR)).compressed()
    groundRegion.shape = (groundRegion.size//3, 3)
    skyRegion = np.ma.array(image, mask=cv2.cvtColor(skyMask, cv2.COLOR_GRAY2BGR)).compressed()
    skyRegion.shape = (skyRegion.size//3, 3)
    
    # calculating covariance matrix of ground and sky regions of image, return covariance value and average RGB values of the two regions
    covarGround, averageGround = cv2.calcCovarMatrix(groundRegion, None, cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)
    covarSky, averageSky = cv2.calcCovarMatrix(skyRegion, None, cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)
    
    #energy optimisation function Jn
    # Jn = 1/γ.|Σs| + |Σg| + γ.|λ1**s| + |λ1**g|
    gamma = 2
    jn= 1 / (
        (gamma * np.linalg.det(covarSky) + np.linalg.det(covarGround)) +
        (gamma * np.linalg.det(np.linalg.eig(covarSky)[1]) +
            np.linalg.det(np.linalg.eig(covarGround)[1])))
    
    return jn

#Calculate optimal sky border position
def OptimalSkyBorderPos(image, gradientImg, minThresh, maxThresh, searchStep):
    
    #initialization
    bOpt = 0
    JnMax = 0
    
    #number of sample points in the search space
    n = ((maxThresh - minThresh) // searchStep) + 1
    
    for k in range(1, n + 1):
        
        #proposed formula only depends on single parameter = t
        t = minThresh + ((maxThresh - minThresh) // n - 1) * (k - 1)
        
        bTmp = CalculateBorder(gradientImg, t) #function2 - current sky border position
        jn = EnergyFunction(bTmp, image) #function3 - calculate energy function

        if jn > JnMax:
            JnMax = jn
            bOpt = bTmp
        
    return bOpt #return the optimal sky border position
    
#post processing to improve segmentation result of masks containing gaps within non-sky region
def PostProcessing(inpMask):
    kernel = np.ones((20,20),np.uint8) * 255
    inv_inpMask = cv2.bitwise_not(inpMask)
    inv_inpMask = cv2.morphologyEx(inv_inpMask, cv2.MORPH_CLOSE, kernel)
    inpMask_1 = cv2.bitwise_not(inv_inpMask)
    return inpMask_1

# Find the average value or brightness of a picture
def AvgBrightness(rgbImage):
    # Convert Image to HSV
    hsv=cv2.cvtColor(rgbImage,cv2.COLOR_RGB2HSV)
    
    # Add up all the pixal values in the V channel  
    sumBrightness=np.sum(hsv[:,:,2])
    area= len(rgbImage)*len(rgbImage[0]) # Area = row * column
    
    # find the average
    avg=sumBrightness/area
    return avg

#Threshold to determine day or night
def Threshold(imageList):
    
    arr = []
    
    count=0
    for image in imageList:
        arr.append(AvgBrightness(image))
        count+=1

    allAvgBrightness = sum(arr) / count
    
    return allAvgBrightness

#Estimate day or night
def EstimateDayNight(rgbImage, imageList):

    # Extract average brightness feature from an RGB image
    avg= AvgBrightness(rgbImage)
    
    # Use the avg brightness feature to predict a label(0,1)
    # 0 = night, 1 = day
    predictedLabel=0
    
    threshold = Threshold(imageList)
    
    if(avg>threshold):
        # if average brightness is above the threshold value ,'We classify it as day'
        predictedLabel=1
        # else Night
    return predictedLabel



#Calculate accuracy of the image
def CalculateAccuracy(mask1, mask2): #(groundtruth, postprocessed mask)
    
    # Ensure both images are binary (0 for black, 1 for white)
     groundTruthBinary = np.where(mask1 == 0, 0, 1)
     predictedBinary = np.where(mask2 == 0, 0, 1)
    
     # Calculate the confusion matrix
     TP = np.sum(np.logical_and(groundTruthBinary == 1, predictedBinary == 1))
     TN = np.sum(np.logical_and(groundTruthBinary == 0, predictedBinary == 0))
     FP = np.sum(np.logical_and(groundTruthBinary == 0, predictedBinary == 1))
     FN = np.sum(np.logical_and(groundTruthBinary == 1, predictedBinary == 0))
    
     # Calculate accuracy
     accuracy = (TP + TN) / (TP + TN + FP + FN)
     return accuracy

#Extract the skyline 
def ExtractSkyline(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask for the skyline
    skylineMask = np.zeros_like(mask)
    
    # Draw the contours of the skyline on the skyline mask
    cv2.drawContours(skylineMask, contours, -1, 255 , thickness=2)
    
    return skylineMask

##Please put Images folder in the same path
##Images folder to contain folders of ['684', '10917', '623', '9708']

#Create Results folder to store output images
##########################################
if os.path.exists("Results"):
    print("Results file exists and is going to be removed.")
    shutil.rmtree("Results")
else:
    print("Results file does not already exist.")
    
try:
    os.makedirs("Results", exist_ok = True)
    print("SUCCESS: The 'Results' directory was created successfully.")
except OSError as error:
    print("ERROR: The 'Results' directory was NOT CREATED successfully.")
print()
##########################################

print("program running")

#Enter sample size from each folders ['684', '10917', '623', '9708']
sample = int(input("Input sample size for the image folders "))

folderList=os.listdir('Images')
print("Obtaining samples from: " , end=" ")
print(folderList)

accuracyList = [] #store the accuracy of all datasets: ['684', '10917', '623', '9708']


for folder in folderList: # folder ['684', '10917', '623', '9708']
    
    #Create folder in Results folder
    try:
        os.makedirs("Results/" + folder, exist_ok = True)
        print("\nSUCCESS: The '"+folder+"' directory was created successfully.")
        
    except OSError as error:
        print("\nERROR: The '"+folder+"' directory was NOT CREATED successfully.")
 
    
    imageDataset = LoadDataset("Images/"+folder+"/", size=50) # load 50 images for processing

    if sample is not None:
        sampleImages = random.sample(imageDataset, sample) #get the sample images
    
    #Convert ground truth mask to grayscale
    groundTruthMask = cv2.imread("Masks/" + folder + ".png") 
    groundTruthMask = cv2.cvtColor(groundTruthMask, cv2.COLOR_BGR2GRAY)
    
    currentAccuracy = [] #accuracy of the current dataset
    
    for img, imageName in sampleImages: #for each image in the sample image
        
        print("=====================")
        print("Folder: " + folder + " Image file name: " + imageName)
        
        #Image processing
        #obtain gradient magnitude image
        gradientImg = CalculateGradientMagnitude(img) 
        #obtain the optimal sky border position
        borderOptimal = OptimalSkyBorderPos(img, gradientImg, 5,600,5) 
        #obtain mask of the sky ground region
        mask = CreateMask(borderOptimal, img)
        #post-processing method is applied and obtain post-processed mask
        postProcessedMask = PostProcessing(mask)
        
        #obtain the skyline of mask
        skyline = ExtractSkyline(postProcessedMask)
            
        #calculate accuracy of predicted mask based on ground truth mask
        accuracy = CalculateAccuracy(groundTruthMask, postProcessedMask) * 100
        print("accuracy: "+ str(round(accuracy, 2)) + "%")
        currentAccuracy.append(accuracy)
        
        #estimate daytime or nighttime
        estimatedaynight = EstimateDayNight(img, (item[0] for item in imageDataset))
        
        if(estimatedaynight == 1):
            print("This image is taken during daytime")
        else: 
            print("This image is taken during nighttime")
        
        #write original image, predicted mask image, predicted skyline image
        cv2.imwrite("Results/"+ folder + "/" + imageName +  "_ori.png", img)
        cv2.imwrite("Results/"+ folder + "/" + imageName +  "_mask.png", postProcessedMask)
        cv2.imwrite( "Results/"+folder+"/"+ imageName + "_skyline.png", skyline)
     
    averageCurrentAccuracy = sum(currentAccuracy) / len(currentAccuracy)
    accuracyList.append(averageCurrentAccuracy)

averageAccuracies = []

for index, folder in enumerate(folderList):
    averageAccuracy = round(accuracyList[index],2)
    print("\naverage accuracy of sample images in " + folder + " is " + str(averageAccuracy) + "%")
    averageAccuracies.append(averageAccuracy)
    
averageAccuracy = round(sum(averageAccuracies) / len(averageAccuracies), 2)
print("\naverage accuracy of all sample images is " + str(averageAccuracy) + "%")
print("\nView the output images in the Results Folder")

cv2.destroyAllWindows()



            

