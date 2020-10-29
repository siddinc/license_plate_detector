import cv2
import os

path = "./test_images/"
for subdir, dirs, files in os.walk(path):
    for file in files:
        re_path = os.path.join(subdir, file)
        image = cv2.imread(re_path)    #read the image
        image = cv2.resize(image, (500, 400))   #resize the image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    #convert image from RGB to grayscale
        filter_image = cv2.bilateralFilter(gray_image, 11, 25, 25)  #noise removal with iterative bilateral filter(removes noise while preserving edges)
        equalized_image = cv2.equalizeHist(filter_image)    #histogram equaliztion increases contrast of image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 2))   #license plate rectangle size array
        morphed_image = cv2.morphologyEx(equalized_image, cv2.MORPH_OPEN, kernel)   #removes noise by performing erosion after dilation
        subtracted_image = cv2.subtract(morphed_image, equalized_image)   #subtract equalized image from morphed image
        _, threshold_image = cv2.threshold(morphed_image, 200, 255, cv2.THRESH_BINARY)  #If pixel value is greater than a threshold value, it is assigned one value (may be white), else it is assigned another value (may be black)
        edged_image = cv2.Canny(threshold_image, 170, 200, L2gradient=True)   #finds edges of the image
        dilation_image = cv2.dilate(edged_image, kernel, iterations = 1)    #increases the white region in the image or size of foreground object increases
        imagex = dilation_image.copy()
        # Find contours based on Edges
        (new, cnts, _) = cv2.findContours(imagex, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
        NumberPlateCnt = None #we currently have no Number plate contour
        
        # loop over our contours to find the best possible approximate contour of number plate
        count = 0
        for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.05 * peri, True)
                if len(approx) == 4:  # Select the contour with 4 corners
                    NumberPlateCnt = approx #This is our approx Number Plate Contour
                    break
        
        cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)     # Drawing the selected contour on the original image
        cv2.imwrite("./output/output-{}.png".format(file[:len(file)-4]), image)
        cv2.imshow("Image", dilation_image)
        cv2.imshow("Imagex", image)     #display the param image in window
        cv2.waitKey(5000)
        cv2.destroyAllWindows()