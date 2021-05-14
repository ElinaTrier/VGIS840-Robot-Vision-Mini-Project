import cv2
import numpy as np
import time

generate_images = True 

# The position of the plane with respect to the world frame
plane_x = 300
plane_y = -400

legoDim = 32 #size of the Lego in millimeter

planeDim = 260 # Height and width of the plane in millimeter

# Mean hue and saturation for colour detection in the HSV colour space
meanColour = {
    "blue":   [200, 250],
    "red":    [0, 250],
    "yellow": [30, 250],
    "orange": [15, 250],
    "green":  [60, 250],
    "white":  [0, 0]
}

# characters ordered colour.
characters = {
    "homer":  ['blue', 'white', 'yellow'],
    "bart":   ['blue', 'red', 'yellow'],
    "marge":  ['green', 'yellow','blue'],
    "lisa":   ['yellow', 'orange', 'yellow'],
    "maggie": ['blue', 'yellow']
}

# arry for lego positions 
legos = []

# array for lego colours
colours = []

def orderPoints(pts):
    #Making an array of coordinates in the order [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]     # the top-left point will have the smallest sum,
    rect[2] = pts[np.argmax(s)]     # the bottom-right point will have the largest sum
    # now, compute the difference between the points
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]  # the top-right point will have the smallest difference,
    rect[3] = pts[np.argmax(diff)]  # the bottom-left will have the largest difference
    
    return rect

def orderCorners( corners ):
    ordered = []
    padding = 10
    last_y = corners[0][1]
    
    corners = sorted(corners , key=lambda y: y[1])
    
    row = []
    
    for i, corner in enumerate(corners):
        
        # Check if y is out of range
        if(corner[1] > last_y - padding  and corner[1] < last_y + padding ):
            # Is within range
            row.append( corner )
            #print(row)
            
        else:

            row = sorted(row , key=lambda x: x[0])
            ordered.extend(row)
            
            row = []
            last_y = corner[1]
            
            row.append( corner )
            
        if( len(corners)-1 == i ):
            row = sorted(row , key=lambda x: x[0])
            ordered.extend(row)
        
    return ordered

def fourPointTransformation(image, pts):
    rect = orderPoints(pts)
    (tl, tr, br, bl) = rect #giving the corners names (topleft,topright,bottomright,bottomleft)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))# maximum distance between bottom-right and bottom-left  x axis
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))#  maximum distance between top-right and top-left  x axis
    maxWidth = max(int(widthA), int(widthB))    #compute the width of the new image, which is the maximum distance on the x axis
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))# maximum distance between the top-right and bottom-right y axis
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2)) # maximum distance between  top-left and bottom-left y axis 
    maxHeight = max(int(heightA), int(heightB))# compute the height of the new image  y axis
    
    
    maxDim = max(maxWidth, maxHeight)# the dimensions of the new image is the maximum distance of the maximum height or width

    dst = np.array([
        [0, 0],
        [maxDim, 0],
        [maxDim, maxDim],
        [0, maxDim]
    ], dtype = "float32")
    
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxDim, maxDim))
    # return the warped image
    return warped, maxDim
    
def removePerspective( image ):
    # Convert to grayscale
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur image a little to remove very sharp edges
    blur = cv2.blur(grayImage,(3,3))
    
    # Convert to binary image (black/white)
    (thresh, binaryImage) = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)
    cv2.imshow ("binary", binaryImage)
    
    # Invert the binary image to find contours
    inverted = cv2.bitwise_not(binaryImage)

    # Find contours in inverted binary image
    (contours, hier) = cv2.findContours(inverted,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    # The first contour is always the most outer contour
    planeContour = contours[0]
       
    if generate_images:
        # Convert back to normal binary
        binaryImage = cv2.bitwise_not(inverted)
        cv2.imwrite('images/binary.jpg', binaryImage)

    # Now fit polyline to outer contour
    epsilon = 5 # This is the maximum distance between the original curve and its approximation (in px).
    points = cv2.approxPolyDP(planeContour, epsilon, True)

    # Generate coordinates for imageTransformation
    coordinates = np.zeros((4, 2))
    for i, point in enumerate(points):
        coordinates[i] = point[0]

    # Warp the image, to get a flat image of the plane = no perspective
    warped, plane_dim_px = fourPointTransformation(image, coordinates)
    
    if generate_images:
        copy = binaryImage.copy()
        new = cv2.cvtColor(copy, cv2.COLOR_GRAY2RGB)
        
        # Draw polyline (all points)
        cv2.drawContours(new, [points], -1, (0, 255, 0), 4)
        
        for point in coordinates:
            cv2.circle(new, (int(point[0]), int(point[1])), 5, (255, 0,0), 3)
        
        cv2.imwrite('images/contour-corners.jpg', new)

    if generate_images:
        cv2.imwrite('images/noPerpective.jpg', warped)
   
    return warped, plane_dim_px


# Map from one variable to another
def pixelMap( x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def findLegePosition( birdEyeImage ):
    # NOW find contours of lego legos on the image with bird eye perspective
    grayImage = cv2.cvtColor(birdEyeImage, cv2.COLOR_BGR2GRAY)
    # Blur image a little to remove very sharp edges
    blur = cv2.blur(grayImage,(5,5))

    # Convert to binary image (black/white)
    (thresh, binaryImage) = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)

    if generate_images:
        cv2.imwrite('images/noPerspectiveBinary.jpg', binaryImage)

    # Erode to get closer to actual border 
    kernel = np.ones((5,5))
    erosion = cv2.erode(binaryImage, kernel,iterations = 1)

    if generate_images:
        inverted = cv2.bitwise_not(erosion)
        cv2.imwrite('images/noPerspectiveBinaryErosion.jpg', inverted)
    
    # Find contours in inverted binary image
    (contours,hier) = cv2.findContours(erosion,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    # Creating empty array
    polygones = []
    corners = []
    h_block = 0
    h_total = 0

    # Now fit polyline to all 16 contours to find locations of lego legos
    for i, contour in enumerate(contours):
        
        epsilon = 0.1 * cv2.arcLength(contours[i], True)
        polygone = cv2.approxPolyDP(contours[i], epsilon, True)#true means it is colsed
        
        new = []
        for point in polygone:
            new.append( [ point[0][0], point[0][1] ] )
        
        # Order points
        new = orderPoints( np.array( new ) )
        new = new.astype(int)
        
        # Save polygone and 1st corner of polygone for later use
        polygones.append(new)
        corners.append(new[0])
     
        h_total = h_total + new[3][1] - new[0][1]
        h_block = h_total / (i+1)
    
    corners = orderCorners(corners)

    if generate_images:
        copy = inverted.copy()
        new = cv2.cvtColor(copy, cv2.COLOR_GRAY2RGB)
        for i, corner in enumerate(corners):
            # cv2.putText(birdEyeImage, str(i), tuple(corner), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(new, tuple(corner), tuple(np.array(corner) + int(h_block)), (0,255,0), 2)
            cv2.circle(new, tuple(corner), 5, (255, 0,0), 3) 
            
        cv2.imwrite('images/legoPosition.jpg', new)

    
    return corners, int(h_block) 


def findLegoColour( birdEyeImage, corners, height ):

    colours = []

    # Loop though all polygones, to determine the colour within this area

    for i, corner in enumerate(corners):
        
        # The first point in the polygone, is the upper left corner
        top = corner

        x = top[0]
        y = top[1]

        # Select area to check
        cropped = birdEyeImage[y:y+height, x:x+height]
        
        # Convert cropped to hsv to check for hue and saturation
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Merge hue and saturation, to detect difference between red and white (has same hue)
        hsv = cv2.merge((h, s))


        for colour in meanColour:
            lower = np.array(meanColour[colour]) - 10
            upper = np.array(meanColour[colour]) + 10

            mask = cv2.inRange(hsv, lower, upper)
            
            if 255 in mask:
                colours.insert(i, colour)
                
    if generate_images:
        new = birdEyeImage.copy()
        new.fill(255)
        for i, point in enumerate(corners):
            
            # Get the colour in HSV
            colour_hsv = meanColour[colours[i]].copy()
            colour_hsv.append(255)
            
            # Convert to RGB
            colour_hsv = np.uint8([[colour_hsv]])
            colour_rgb = cv2.cvtColor(colour_hsv, cv2.COLOR_HSV2BGR)[0][0]
            
            colour = tuple(colour_rgb)
            
            cv2.rectangle(new, tuple(point), tuple(np.array(point) + int(height)), (int(colour[0]), int(colour[1]), int(colour[2])), -1)
            cv2.rectangle(new, tuple(point), tuple(np.array(point) + int(height)), (0,0,0), 1)
            cv2.circle(new, tuple(point), 5, (255, 0,0), 3) 
            
        cv2.imwrite('images/LegoColours.jpg', new)

                
    return colours


def run( name ):

    # Load image taken from roboDK 2d camera
    filename = 'images/original.jpg'
    image = cv2.imread(filename)

    # Remove perspective from camera image, and get the size of the square in pixels for later use
    birdEyeImage, plane_dim_px = removePerspective(image)
    
    # Detect legos and get the mean height of the legos in pixels
    legos, block_height = findLegePosition( birdEyeImage )
    
    # Detect the colour of each lego
    colours = findLegoColour( birdEyeImage, legos, block_height )

    # Get the recipe for "name" in the big recipe-book.
    recipe = characters[name]
        
    coordinates_px = []
    
    # Loop though the recipe, and find the legos needed
    for item in recipe:
        
        # Get index of the first coloured lego in the list
        index = colours.index(item)
        
        # Set the lego to "used" such that it cannot be used any more
        colours[index] = '-'
        
        coordinates_px.append( legos[index] )
        
    
    coordinates_mm = []
    
    for i, coordinate in enumerate(coordinates_px):
        
        x = coordinate[0]
        y = coordinate[1]
        
        x_mm = plane_x + pixelMap(x, 0, plane_dim_px, planeDim, 0) - legoDim/2
        y_mm = plane_y + pixelMap(y, 0, plane_dim_px, 0, planeDim) + legoDim/2
        
        coordinates_mm.append([x_mm, y_mm])
        
    return coordinates_mm

run( 'marge' )
