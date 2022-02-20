import cv2
import numpy as np
import random
import copy
import time
import math
from PIL import Image, ImageChops

class Painting:

    def __init__(self, img_path, oldMutation, mutationStrength):
        self.original_img = Image.open(img_path)
        self.img_grey = self.original_img.convert("L")

        # Load brushes
        self.maxBrushNumber = 4
        self.brushes = self.preload_brushes('brushes/watercolor/', self.maxBrushNumber)

        # Stroke boundaries
        self.bound = self.img_grey.size
        self.minSize = 0.1
        self.maxSize = 0.7
        self.brushSide = 300  # brush image resolution in pixels
        self.padding = int(self.brushSide*self.maxSize / 2 + 25)

        # Strokes and current painting
        self.strokes = []
        self.canvas_memory = []
        self.current_error = self.bound[0] * self.bound[1] * 255

        # Mutation properties
        self.oldMutation = oldMutation
        self.mutationStrength = mutationStrength

        # PPA properties
        self.fitness = None
        self.norm_fitness = None
        self.MSE_calced = False
        self.mutateCount = 1
        self.cycles_alive = 0

        # SA properties
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
            # imgs.append(Image.open(path + str(i+1) +'.jpg'))
            brushImg = Image.open(path + str(i+1) +'.jpg')
            imgs.append(brushImg.convert("L"))
            # imgs.append(cv2.imread(path + str(i+1) +'.jpg'))
        return imgs

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
            if self.oldMutation:
                mutatedStroke.color = mutatedStroke.new_color()
            else:
                mutatedStroke.color = mutatedStroke.mut_color(mutatedStroke.color, self.mutationStrength)
        # mutate size
        elif mutateOption == 2:
            if self.oldMutation:
                mutatedStroke.size = mutatedStroke.new_size(self.minSize, self.maxSize)
            else:
                mutatedStroke.size = mutatedStroke.mut_size(self.minSize, self.maxSize, mutatedStroke.size, self.mutationStrength)
        # mutate position
        elif mutateOption == 3:
            if self.oldMutation:
                mutatedStroke.posY, mutatedStroke.posX = mutatedStroke.gen_new_positions(self.bound)
            else:
                mutatedStroke.posY, mutatedStroke.posX = mutatedStroke.mut_positions(self.bound, [mutatedStroke.posY, mutatedStroke.posX], self.mutationStrength)
        # mutate rotation
        elif mutateOption == 4:
            if self.oldMutation:
                mutatedStroke.rotation = mutatedStroke.new_rotation()
            else:
                mutatedStroke.rotation = mutatedStroke.mut_rotation(mutatedStroke.rotation, self.mutationStrength)
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
        err = np.sum((np.array(imageA).astype(float) - np.array(imageB).astype(float)) ** 2)
        err /= float(imageA.size[0] * imageA.size[1])

        # return the MSE, the lower the error, the more "similar" the two images are
        return err

    def draw(self, strokes):
        # set image to pre generated and apply padding
        p = self.padding
        inImg = Image.new('RGBA', (self.bound[0]+2*p, self.bound[1]+2*p), (0, 0, 0))

        # draw every all brush strokes
        for i in range(len(strokes)):
            inImg = self.__drawStroke(strokes[i], inImg)

        y = inImg.size[0]
        x = inImg.size[1]
        # remove padding
        inImg = inImg.crop((p, p, y-p, x-p))
        return inImg

    def __drawStroke(self, stroke, inImg):
        # get stroke data
        color = stroke.color
        posX = int(stroke.posX) + self.padding  # add padding since indices have shifted
        posY = int(stroke.posY) + self.padding
        # print(str(posX) + " " + str(posY))
        size = stroke.size
        rotation = stroke.rotation
        brushNumber = int(stroke.brush_type)

        # load brush alpha
        brushImg = self.brushes[brushNumber]
        # resize the brush
        brushImg = brushImg.resize((math.ceil(brushImg.size[0]*size),math.ceil(brushImg.size[1]*size)), resample=Image.BICUBIC, box=None, reducing_gap=1.01)
        # rotate
        brushImg = brushImg.rotate(rotation)

        # create a colored canvas
        rows, cols = brushImg.size
        foreground = Image.new('RGBA', (rows, cols), (color[0], color[1], color[2]))

        # find ROI
        inImg_rows, inImg_cols = inImg.size
        y_min = int(posY - rows/2)
        y_max = int(posY + (rows - rows/2))
        x_min = int(posX - cols/2)
        x_max = int(posX + (cols - cols/2))

        background = inImg.crop((x_min, y_min, x_max, y_max))  # get ROI
        alpha = brushImg

        try:
            foreground.putalpha(alpha)

            # Multiply the background with ( 1 - alpha )
            alpha_inv = ImageChops.invert(alpha)
            background.putalpha(alpha_inv)

            # Add the masked foreground and background.
            background.alpha_composite(foreground, dest=(0, 0), source=(0, 0))
            inImg.alpha_composite(background, dest=(x_min, y_min), source=(0, 0))

        except(Exception):
            print('------ \n', 'in image ',inImg.size)
            print('pivot: ', posY, posX)
            print('brush size: ', self.brushSide)
            print('brush shape: ', brushImg.size)
            # print(" Y range: ", rangeY, 'X range: ', rangeX)
            # print('bg coord: ', posY, posY+rangeY, posX, posX+rangeX)
            print('fg: ', foreground.size)
            print('bg: ', background.size)
            print('alpha: ', alpha.size)

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
        posX = int(random.randrange(0, bound[0]))
        posY = int(random.randrange(0, bound[1]))
        return [posY, posX]

    def mut_color(self, color, mutationStrength):
        new_color = color
        mu = 0
        sigma = mutationStrength * 255
        new_color[0] += math.ceil(np.random.normal(mu, sigma))
        new_color[1] += math.ceil(np.random.normal(mu, sigma))
        new_color[2] += math.ceil(np.random.normal(mu, sigma))
        if new_color[0] < 0:
            new_color[0] = 0
        if new_color[1] < 0:
            new_color[1] = 0
        if new_color[2] < 0:
            new_color[2] = 0
        if new_color[0] > 255:
            new_color[0] = 255
        if new_color[1] > 255:
            new_color[1] = 255
        if new_color[2] > 255:
            new_color[2] = 255
        return new_color

    def mut_size(self, minSize, maxSize, size, mutationStrength):
        mu = 0
        sigma = mutationStrength*(maxSize-minSize)
        mutation = np.random.normal(mu, sigma)
        new_size = size + mutation
        if new_size > maxSize:
            new_size = maxSize
        if new_size < minSize:
            new_size = minSize
        return new_size

    def mut_rotation(self, rotation, mutationStrength):
        mu = 0
        sigma = mutationStrength * 360
        mutation = np.random.normal(mu, sigma)
        rotation += mutation
        if rotation < -180:
            rotation = -180
        if rotation > 180:
            rotation = 180
        return rotation

    def mut_brush_type(self, maxBrushNumber):
        new_brush_type = random.randrange(1, maxBrushNumber)
        return new_brush_type

    def mut_positions(self, bound, position, mutationStrength):
        mu = 0
        position[0] = position[0]+np.random.normal(mu, mutationStrength * bound[0])
        position[1] = position[1]+np.random.normal(mu, mutationStrength * bound[1])
        if position[0] < 0:
            position[0] = 0
        if position[1] < 0:
            position[1] = 0
        if position[0] > bound[0]:
            position[0] = bound[0]
        if position[1] > bound[1]:
            position[1] = bound[1]
        return position

    def randomAttributes(self, minSize, maxSize, maxBrushNumber, bound):
        self.color = self.new_color()
        self.size = self.new_size(minSize, maxSize)
        self.posY, self.posX = self.gen_new_positions(bound)
        self.rotation = self.new_rotation()
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
