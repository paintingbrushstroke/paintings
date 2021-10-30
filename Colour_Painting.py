import cv2
import numpy as np
import os
import random
#import matplotlib.pyplot as plt
import copy
import time



class Painting:

    def __init__(self, img_path):
        self.original_img = cv2.imread(img_path)

        self.img_grey = cv2.cvtColor(self.original_img,cv2.COLOR_BGR2GRAY)
        self.img_grads = self._imgGradient(self.img_grey)

        #Load brushes
        self.maxBrushNumber = 5
        self.brushes = self.preload_brushes('brushes/watercolor/', self.maxBrushNumber)

        #Stroke boundaries
        self.bound = self.img_grey.shape
        self.minSize = 0.1 #0.1 #0.3
        self.maxSize = 0.7 #0.3 # 0.7
        self.brushSide = 300 #brush image resolution in pixels
        self.padding = int(self.brushSide*self.maxSize / 2 + 5)
        
        # Strokes and current painting
        self.strokes = []
        self.canvas_memory = []
        self.current_error = self.bound[0] * self.bound[1] * 255

        # PPA properties
        self.fitness = None
        self.norm_fitness = None
        self.MSE_calced = False
        self.mutateCount = 1
        self.cycles_alive = 0

        #SA properties
        self.current_best_error = self.bound[0] * self.bound[1] * 255
        self.current_best_canvas = []

    # Initialize the random brush strokes for a stroke count
    def init_strokes(self, stroke_count):
        for index in range(0, stroke_count):
            brush = Brush_stroke(index)
            brush.randomAttributes(self.minSize, self.maxSize, self.maxBrushNumber, self.bound)
            self.strokes.append(brush)

    def preload_brushes(self, path, maxBrushNumber):
        imgs = []
        for i in range(maxBrushNumber):
            imgs.append(cv2.imread(path + str(i) +'.jpg'))
            # imgs.append(cv2.imread(path + str(i) +'.jpg'))
        return imgs


    def _imgGradient(self, img):
        #convert to 0 to 1 float representation
        img = np.float32(img) / 255.0 
        # Calculate gradient 
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        # Python Calculate gradient magnitude and direction ( in degrees ) 
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        #normalize magnitudes
        mag /= np.max(mag)
        #lower contrast
        mag = np.power(mag, 0.3)
        return mag, angle

    def evolve_strokes(self, evaluations, filename):
        logger  = "log-" + filename + "-" + str(len(self.strokes)) + "-" + str(evaluations)
        f = open(logger, "w")
        for i in range(evaluations):
            mutatedStrokes = self.mutate()

            # calculate error
            error, img = self.calcError(mutatedStrokes)

            # if the error is lowered set the mutated version as the new version
            if error < self.current_error:
                self.current_error = error
                self.canvas_memory.append(img)
                self.strokes = mutatedStrokes
                # print(i, ". new best:", self.current_error)
                f.write(str(i+1) + "," + str(self.current_error) + "\n")

            # output image every 100.000
            # if i%3000 == 0 or i == 0:
            #     cv2.imwrite(filename + "-" + str(len(self.strokes)) + "-" + str(i) + ".png" , self.canvas_memory[-1])

    def mutate(self):
        # copy strokes and stroke object to make reverting possible
        copyStrokes = copy.deepcopy(self.strokes)
        mutatedStroke = random.choice(copyStrokes)

        options = [0,1,2,3,4,5]
        mutateOption = random.choice(options)
        # swap index with other brush
        if mutateOption == 0:
            # remove swapped brushStroke form strokes
            copyStrokes.pop(mutatedStroke.index)
    
            # select random number for the insert
            insertIndex = int(random.randrange(0, len(copyStrokes)))
            copyStrokes.insert(insertIndex, mutatedStroke)

            # fix all indeces
            for i in range(0, len(copyStrokes)):
                copyStrokes[i].index = i

        # mutate color
        elif mutateOption == 1:
            mutatedStroke.color = mutatedStroke.new_color()
        # mutate size
        elif mutateOption == 2:
            mutatedStroke.size = mutatedStroke.new_size(self.minSize, self.maxSize)
        # mutate position
        elif mutateOption == 3:
            mutatedStroke.posY, mutatedStroke.posX = mutatedStroke.gen_new_positions(self.bound)
        # mutate rotation
        elif mutateOption == 4:
            mutatedStroke.rotation = mutatedStroke.new_rotation()
        # mutate brush type
        elif mutateOption == 5:
            mutatedStroke.brush_type = mutatedStroke.new_brush_type(self.maxBrushNumber)

        return copyStrokes      

    # Draw all strokes and calculate the error between the strokes and real img
    def calcError(self, newStrokes):
        myImg = self.draw(newStrokes)

        error = self.mse(self.original_img, myImg)
        return (error, myImg)


    def mse(self, imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        
        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err      
        
    def draw(self, newStrokes):
        myImg = self.drawAll(newStrokes)
        return myImg
        
    def drawAll(self, strokes):
        #set image to pre generated
        inImg = np.zeros((self.bound[0], self.bound[1], 3), np.uint8)

        #apply padding
        p = self.padding
        inImg = cv2.copyMakeBorder(inImg, p,p,p,p,cv2.BORDER_CONSTANT,value=[0,0,0])
        #draw every all brush strokes
        for i in range(len(strokes)):
            inImg = self.__drawStroke(strokes[i], inImg)
        #remove padding
        y = inImg.shape[0]
        x = inImg.shape[1]
        return inImg[p:(y-p), p:(x-p)]       
        
    def __drawStroke(self, stroke, inImg):
        #get stroke data
        color = stroke.color
        posX = int(stroke.posX) + self.padding #add padding since indices have shifted
        posY = int(stroke.posY) + self.padding
        size = stroke.size
        rotation = stroke.rotation
        brushNumber = int(stroke.brush_type)

        #load brush alpha
        brushImg = self.brushes[brushNumber]
        #resize the brush
        brushImg = cv2.resize(brushImg,None,fx=size, fy=size, interpolation = cv2.INTER_CUBIC)
        #rotate
        brushImg = self.__rotateImg(brushImg, rotation)
        #brush img data
        rows, cols, _ = brushImg.shape
        
        #create a colored canvas
        myClr = np.full((rows, cols,3), color)
        # myClr[:, :] = color

        #find ROI
        inImg_rows, inImg_cols, _ = inImg.shape
        y_min = int(posY - rows/2)
        y_max = int(posY + (rows - rows/2))
        x_min = int(posX - cols/2)
        x_max = int(posX + (cols - cols/2))
        
        # Convert uint8 to float
        foreground = myClr[0:rows, 0:cols].astype(float)
        
        background = inImg[y_min:y_max,x_min:x_max].astype(float) #get ROI
        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha = brushImg.astype(float)/255.0

        try:
            # Multiply the foreground with the alpha matte
            foreground = cv2.multiply(alpha, foreground)
            
            
            # Multiply the background with ( 1 - alpha )
            background = cv2.multiply(np.clip((1.0 - alpha), 0.0, 1.0), background)
            # Add the masked foreground and background.
            outImage = (np.clip(cv2.add(foreground, background), 0.0, 255.0)).astype(np.uint8)

            
            inImg[y_min:y_max, x_min:x_max] = outImage
        except:
            print('------ \n', 'in image ',inImg.shape)
            print('pivot: ', posY, posX)
            print('brush size: ', self.brushSide)
            print('brush shape: ', brushImg.shape)
            print(" Y range: ", rangeY, 'X range: ', rangeX)
            print('bg coord: ', posY, posY+rangeY, posX, posX+rangeX)
            print('fg: ', foreground.shape)
            print('bg: ', background.shape)
            print('alpha: ', alpha.shape)
        
        return inImg

    def __rotateImg(self, img, angle):
        rows,cols,channels = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        return dst
    
class Brush_stroke:
     
    def __init__(self, index):


        self.index = index
        self.color = [0,0,0]
        self.size = 0
        self.posY, self.posX = [0,0]
        self.rotation = 0
        self.brush_type = 1

    def new_color(self):
        new_color = [random.randrange(0, 255),random.randrange(0, 255),random.randrange(0, 255)]
        return new_color

    def new_size(self, minSize, maxSize):
        new_size = random.random()*(maxSize-minSize) + minSize
        return new_size
    
    def new_rotation(self):
        new_rotation = random.randrange(-180, 180)
        return new_rotation
    
    def new_brush_type(self, maxBrushNumber):
        new_brush_type = random.randrange(1, maxBrushNumber)
        return new_brush_type


    # generate new brush positions for the brush
    def gen_new_positions(self, bound):
        posY = int(random.randrange(0, bound[0]))
        posX = int(random.randrange(0, bound[1]))
        return [posY, posX]

    def randomAttributes(self, minSize, maxSize, maxBrushNumber, bound):
        #random color
        self.color = self.new_color()
        #random size
        self.size = self.new_size(minSize, maxSize)
        #random pos
        self.posY, self.posX = self.gen_new_positions(bound)
        #random rotation
        self.rotation = self.new_rotation()
        #random brush number
        self.brush_type = self.new_brush_type(maxBrushNumber)


if __name__ == "__main__":
    t = time.localtime()
    old_time = time.strftime("%H:%M:%S", t)
    print(old_time)

    evolve = Painting("imgs/mona.png")
    evolve.init_strokes(200)

    evolve.evolve_strokes(100000, "mona.png")

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(old_time, current_time)
    
    plt.imshow(evolve.canvas_memory[-1], cmap='gray')
    plt.show()
                    
