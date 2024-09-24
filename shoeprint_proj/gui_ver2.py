import pickle, os, cv2, gzip
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from PIL import Image, ImageTk
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient
from skimage.feature import local_binary_pattern



# loading all web scraped images 
#images_folder_path = "C:\\Users\\gichu\\Documents\\shoeprint_proj\\final_dataset\\1"

folder_name = "3"  # Replace this with the folder name containing the images
images_folder_path = os.path.join("C:\\Users\\gichu\\Documents\\shoeprint_proj\\final_dataset", folder_name)

all_files = os.listdir(images_folder_path)
all_files_len = len(all_files)

# connecting to db using mongo client
load_dotenv(find_dotenv()) # load env var

#password = os.environ.get("MONGO_PWD")
#connection_string = "mongodb+srv://gichurud02:{}@shoeprintcluster.ibkpwhz.mongodb.net/?retryWrites=true&w=majority".format(password)

password = os.environ.get("MONGO_PWD_2")
connection_string = "mongodb+srv://gitagamad02:ZJwH1czYMeo85fvW@cluster0.5butm9b.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0".format(password)
client = MongoClient(connection_string)
test_db = client.test # accesing the test db

image_counter = 0
n = 1000
sift = cv2.SIFT_create(n) # limiting to n kps
orb = cv2.ORB_create(n)

# purpose of func is to rotate the image in replication of shoeprints in diff orientations
def rotation_aug(img, angle = 15):
    rows, cols = img.shape[:2]
    center = (cols/2, rows/2) # center of rotation
    rotation_matrix = cv2.getRotationMatrix2D(center , angle, 1) # creates rotation matrix
    rotated_img = cv2.warpAffine(img, rotation_matrix, (cols, rows)) # applies rotation
    return rotated_img

# purose of this function is to erode parts of the image to replicate partial footprints
def salt_and_pepper_noise(img):
    L, W = img.shape[0], img.shape[1]
    noise =np.zeros((L, W),dtype=np.uint8) # creating white background of same size
    cv2.randu(noise,0,255) # fills noise with rand vals with uniform dis that can be thresholded
    noise = cv2.threshold(noise,185,255,cv2.THRESH_BINARY)[1]
    return cv2.add(img,noise) # noisy image

# need to alter this so i find a random cop around the center to reduce likelyhood of empty crop
def cropped_aug(img, crop_len = 80 , crop_width = 80):
    CL, CW = crop_len, crop_width 
    L, W = img.shape[0], img.shape[1]

    # starting from 60 instead of 0. meaning crop is more likely to be cenetered
    start_row = np.random.randint(60, L - CL)
    start_col = np.random.randint(60, W - CW)
    return image[start_row:start_row+CL, start_col:start_col+CW]


def convert_approach_one(image, thresh=0, max_val=255):
      conv_to_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # first convert the output to grayscale
      accentuate = cv2.dilate(conv_to_greyscale, np.ones((3,3), dtype=np.uint8) ) # accentuate lines
      erode = cv2.erode(accentuate, np.ones((2,2))) # erosion, to further define gaps in sole patterns
      smoothen = cv2.GaussianBlur(erode, (7, 7), 1) # blur to smoothen affects of erosion
      thresh, blacknwhite = cv2.threshold(smoothen, thresh, max_val, cv2.THRESH_BINARY) #
      invert = cv2.bitwise_not(blacknwhite)
      return invert


def update_label(value):
    global scale_val
    scale_val = value # need to set the curr thresh val
    new_thresh = convert_approach_one(np_image, int(value), 255)
    new_thresh = rescaled_img(new_thresh, 600)
    new_image = ImageTk.PhotoImage(Image.fromarray(new_thresh))
    image_label.config(image=new_image)
    image_label.image = new_image


def apply_augmentations(img):    
    rotated_image = rotation_aug(img, 30) # rotation aug
    eroded_image = salt_and_pepper_noise(img)
    #cropped_image = cropped_aug(img)
    #print(rotated_image.shape, eroded_image.shape)
    return rotated_image, eroded_image #cropped_image


def sift_feature(sift, img):
    kp , desc = sift.detectAndCompute(img, None)
    img_with_keypoints = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Plot the image with keypoints
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    return desc #flatten to 1D


def orb_feature(orb, img):
    kp , desc = orb.detectAndCompute(img, None)
    return desc # return or desc of shape (n,32)


def lbp(img, n, r, meth):
    lbp =  local_binary_pattern(img, n, r, meth)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n + 3), range=[0, n + 3])
    hist = hist.astype("float")
    hist /= np.sum(hist)  # normalize to max of 1

    # Plot the LBP histogram
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(hist)), hist, width=0.8, color='gray')
    plt.xlabel('LBP pixel intensity')
    plt.ylabel('Frequency')
    plt.title('Local Binary Pattern (LBP) Histogram')
    plt.show()

    return lbp


def save_image(image, path):
    print("the image has beeen saved!!! :)" , path)
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)


# Inverted variant of original function, to handle relative pixel intensities
def append_data_inverted(feature_alg=sift_feature, feature=sift, name="sift"):

    # ~~~~~~~~~~~~~~~ FINAL PREPROCESSING ~~~~~~~~~~~~~~~~~~~~
    global np_image
    global image_index
    global image_counter

    brand, version, x = all_files[image_index].split("_")
    print(brand, version, x)

    # cropping thresholded image using border box
    accepted_thresh = convert_approach_one(np_image, int(scale_val), 255) # thresh using accepted val
    invert_thresh =  cv2.bitwise_not(accepted_thresh)
    coords = cv2.findNonZero(invert_thresh)
    x,y,w,h = cv2.boundingRect(coords) 
    inverted_mask = cv2.bitwise_not(accepted_thresh[y:y+h, x:x+w])

    """this cropped var is to crop the black and white image but lbp and sift work better on greyscale so gonna crop that instead"""
    #cropped = accepted_thresh[y:y+h, x:x+w] # Crop according to coords
    crop = np_image[y:y+h, x:x+w]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) # conv image to greyscale for feature extraction

    # NOT SAVING FOR NOW save_image(crop, f"C:\\Users\\gichu\\Documents\\shoeprint_proj\\new_unaltered\\{brand}_{version}_{image_counter}.png") # saving the unaltered image also
    
    #save_image(inverted_mask, f"C:\\Users\\gichu\\Documents\\shoeprint_proj\\masks\\{brand}_{version}_{image_counter}.png")
    zop = cv2.bitwise_and(crop, crop, mask=inverted_mask)
    cropped = crop

    # resize to consitent width 
    # only issus is heights may differ as we cannot alter both and expect to maintain same ratio
    desired_width = 400
    new_width, new_height = cropped.shape[1], cropped.shape[0]
    
    if new_width == 0: # empty image threfore do nothing
        return

    wpercent = (desired_width / float(new_width))
    hsize = int((float(new_height) * float(wpercent)))
    cropped = cv2.resize(cropped, (desired_width, hsize))

    # pad images to constant dimensions with val 0 if the image is below desired height
    desired_height = 150
    if cropped.shape[0] < desired_height:
        pad_top = abs(cropped.shape[0] - desired_height)
        cropped = np.pad(cropped, ((pad_top, 0), (0, 0)), mode='constant', constant_values=0)


    # feature extraction
    rotated_image, eroded_image = apply_augmentations(cropped)
    all_augmentations = [cropped, rotated_image, eroded_image, zop]
    transformation = ["normal", "rotation", "eroded", "mask"]

    # Plot the augmented images
    fig, axes = plt.subplots(1, len(all_augmentations), figsize=(15, 5))  # Adjust figsize as needed

    for i, (image, transformation_name) in enumerate(zip(all_augmentations, transformation)):
        axes[i].imshow(image, cmap='gray')  # Assuming images are grayscale, adjust cmap as needed
        axes[i].set_title(transformation_name)
        axes[i].axis('off')

    plt.show()

    n, r, meth = 16, 3, "uniform"
    
    all_alg_descriptors = [feature_alg(feature, images) for images in all_augmentations]
    all_lbp_descriptors = [lbp(images, n, r, meth) for images in all_augmentations] # lbp for whole image
    #all_mblbp_descriptors = [mb_lbp(images, images.shape[0], images.shape[1], 10, 10) for images in all_augmentations] # concatenated lbp


    ##for i in range(len(transformation)):
    ##   save_image(all_augmentations[i], f"C:\\Users\\gichu\\Documents\\shoeprint_proj\\new_cnn_image_dataset\\{brand}_{version}_{transformation[i]}_{image_counter}.png")
    ##    image_counter += 1

    # appending data to database for all transformations 
    collection = test_db.descriptors # accesing test collection

    # need to compress to save storage space

    
    #"mb-lbp": all_mblbp_descriptors[i].tolist(),
    for i in range(0, len(all_augmentations)):    
        serialize_lbp = pickle.dumps(all_lbp_descriptors[i].tolist())
        serialize_sift = pickle.dumps( all_alg_descriptors[i].tolist())

        compressed_lbp = gzip.compress(serialize_lbp)
        compressed_sift = gzip.compress(serialize_sift)


        shoeprint_data = {
            "brand" : brand,
            "version": version,
            "lbp": compressed_lbp,
            f"{name}": compressed_sift,
            "image_path": images_folder_path + f"\\{brand}_{version}_{x}",
            "transformation" : transformation[i]
            }
        collection.insert_one(shoeprint_data) # insery data into collection

def append_data(feature_alg=sift_feature, feature=sift, name="sift"):

    # ~~~~~~~~~~~~~~~ FINAL PREPROCESSING ~~~~~~~~~~~~~~~~~~~~
    global np_image
    global image_index
    global image_counter

    brand, version, x = all_files[image_index].split("_")
    print(brand, version, x)
    
    # cropping thresholded image using border box
    accepted_thresh = convert_approach_one(np_image, int(scale_val), 255) # thresh using accepted val
    coords = cv2.findNonZero(accepted_thresh)
    x,y,w,h = cv2.boundingRect(coords)
    inverted_mask = cv2.bitwise_not(accepted_thresh[y:y+h, x:x+w])

    """this cropped var is to crop the black and white image but lbp and sift work better on greyscale so gonna crop that instead"""
    #cropped = accepted_thresh[y:y+h, x:x+w] # Crop according to coords
    crop = np_image[y:y+h, x:x+w]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) # conv image to greyscale for feature extraction

    ##save_image(crop, f"C:\\Users\\gichu\\Documents\\shoeprint_proj\\new_unaltered\\{brand}_{version}_{image_counter}.png")
    #save_image(inverted_mask, f"C:\\Users\\gichu\\Documents\\shoeprint_proj\\masks\\{brand}_{version}_{image_counter}.png")
    zop = cv2.bitwise_and(crop, crop, mask=inverted_mask)
    cropped = crop

    # resize to consitent width 
    # only issus is heights may differ as we cannot alter both and expect to maintain same ratio
    desired_width = 400
    new_width, new_height = cropped.shape[1], cropped.shape[0]
    
    if new_width == 0: # empty image threfore do nothing
        return

    wpercent = (desired_width / float(new_width))
    hsize = int((float(new_height) * float(wpercent)))
    cropped = cv2.resize(cropped, (desired_width, hsize))

    # pad images to constant dimensions with val 0 if the image is below desired height
    desired_height = 150
    if cropped.shape[0] <= desired_height:
        pad_top = abs(cropped.shape[0] - desired_height)
        cropped = np.pad(cropped, ((pad_top, 0), (0, 0)), mode='constant', constant_values=255)

    # just crop the image to the size we want, should try alternatively ro resize
    if cropped.shape[0] > desired_height:
        cropped = cropped[:150, :400]


    # feature extraction
    rotated_image, eroded_image = apply_augmentations(cropped)
    all_augmentations = [cropped, rotated_image, eroded_image, zop]
    transformation = ["normal", "rotation", "eroded", "mask"]

    # Plot the augmented images
    fig, axes = plt.subplots(1, len(all_augmentations), figsize=(15, 5))  # Adjust figsize as needed

    for i, (image, transformation_name) in enumerate(zip(all_augmentations, transformation)):
        axes[i].imshow(image, cmap='gray')  # Assuming images are grayscale, adjust cmap as needed
        axes[i].set_title(transformation_name)
        axes[i].axis('off')

    plt.show()

    n, r, meth = 8, 1, "uniform"
    
    all_alg_descriptors = [feature_alg(feature, images) for images in all_augmentations]
    all_lbp_descriptors = [lbp(images, n, r, meth) for images in all_augmentations] # lbp for whole image
    #all_mblbp_descriptors = [mb_lbp(images, images.shape[0], images.shape[1], 10, 10) for images in all_augmentations] # concatenated lbp

    ##for i in range(len(transformation)):
    ##    save_image(all_augmentations[i], f"C:\\Users\\gichu\\Documents\\shoeprint_proj\\new_cnn_image_dataset\\{brand}_{version}_{transformation[i]}_{image_counter}.png")
    ##    image_counter += 1

    

    #flat_lbpsift = Binary(pickle.dumps(concat_desc.tolist(), protocol=2))
    # appending data to database for all transformations 
    collection = test_db.descriptors # accesing test collection
    
    for i in range(0, len(all_augmentations)):   

        serialize_lbp = pickle.dumps(all_lbp_descriptors[i].tolist())
        serialize_sift = pickle.dumps( all_alg_descriptors[i].tolist())

        compressed_lbp = gzip.compress(serialize_lbp)
        compressed_sift = gzip.compress(serialize_sift)

        shoeprint_data = {
            "brand" : brand,
            "version": version,
            "lbp": compressed_lbp,
            f"{name}": compressed_sift,
            "image_path": images_folder_path + f"\\{brand}_{version}_{x}",
            "transformation" : transformation[i]
            }
        
        collection.insert_one(shoeprint_data) # insery data into collection

    # #"mb-lbp": all_mblbp_descriptors[i].tolist(),

def next_image():
    global image_index
    global np_image

    image_index += 1
    if image_index >= all_files_len: # if exhausted list of all images
        return

    # UPDATE ORINGAL IMAGE DISPLAY
    image = Image.open(images_folder_path + "\\" + all_files[image_index])
    np_image = np.array(image)

    image.thumbnail((300, 600)) # resize img to fit frame
    init = ImageTk.PhotoImage(image) 
    original_image_label.config(image=init ) # update with new img
    original_image_label.image = init

    # UPDATE IMAGE THRESHOLD IMAGE DISPLAY

    ### MAKE INIITIAL THRESH VAL == OTSU VAL ####
    temp_img = np.array(image.convert('L'))
    init_thresh_val, _ = cv2.threshold(temp_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    original_thresholded_img = convert_approach_one(np_image, init_thresh_val, 255) # thresh init image to be displayed by default
    original_thresholded_img = rescaled_img(original_thresholded_img, 600)
    original_thresholded_img = ImageTk.PhotoImage(Image.fromarray(original_thresholded_img)) # conv back from np.array to photo obj

    image_label.config(image=original_thresholded_img)
    image_label.image = original_thresholded_img
    slider.set(init_thresh_val)


# rescale image whilst maintaining aspect ration
def rescaled_img(image, desired_width = 600):
    new_width, new_height = image.shape[1], image.shape[0]
    wpercent = (desired_width / float(new_width))
    hsize = int((float(new_height) * float(wpercent)))
    image = cv2.resize(image, (desired_width, hsize))
    return image

# defining num of squares width-wise and length-wise with constant dimensions
# https://www.mdpi.com/2073-8994/13/2/296
def mb_lbp(img, img_w, img_h, length_squares_num, width_squares_num): 
    n, r, method = 8, 1, "uniform"
    rows, cols = length_squares_num, width_squares_num
    square_height, square_width = int(img_h / length_squares_num) ,  int(img_w / width_squares_num)

    lbp_vectors = []

    for i in range(rows):
        for j in range(cols):
           
            y_cord, end_y_cord = int(i * square_height), int((i + 1) * square_height) # row
            x_cord, end_x_cord  = int(j * square_width), int((j + 1) * square_width) # col
            square_region = img[x_cord:end_x_cord, y_cord:end_y_cord]
            lbp_region = lbp(square_region, n, r, method) # calc lbp for square region
            lbp_vectors.append(lbp_region)
    
    running_complete_lbp = lbp_vectors[0]
    for i in range(1, len(lbp_vectors)):
        running_complete_lbp += lbp_vectors[i]
    
    running_complete_lbp /= np.sum(running_complete_lbp)

    """
    print(running_complete_lbp)
    plt.bar(range(len(running_complete_lbp)), running_complete_lbp)
    plt.title("Combined Histogram")
    plt.xlabel("LBP Code")
    plt.ylabel("Frequency")
    plt.show()
    """
    return running_complete_lbp

### testing region
##test_img = Image.open('C:\\Users\\gichu\\Documents\\shoeprint_proj\\images\\Jordan_2_5.jpg')  # Replace with the actual filename
#test_img = Image.open('C:\\Users\\gichu\Documents\\shoeprint_proj\\saved_image5.png')  # Replace with the actual filename
#test_img_array = np.array(test_img)
##print(test_img_array.shape)
#mmb_lbp(test_img_array, test_img_array.shape[0], test_img_array.shape[1], 3, 3) #####

root = Tk()
root.title("Dataset creator")
root.rowconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
scale_val = 0

image_index = 0
image_path = images_folder_path + "\\" + all_files[image_index]

image = Image.open(image_path)
np_image = np.array(image)

#------------------------------ LEFT FRAME --------------------------------
# containing original image, and buttons to gen new image or add to dataset

frame_w, frame_h = 300, 600
image.thumbnail((frame_w, frame_h)) # resize img to fit frame

left_frame = Frame(root, width=200, height=400)
left_frame.grid(row=0, column=0, padx=10, pady=5)

init_img = ImageTk.PhotoImage(image) # conv to photoimage obj
original_image_label = Label(left_frame, image=init_img)
original_image_label.pack(pady=10)

button1 = Button(left_frame, text="Add", command=append_data,width=15, height=2).pack() # button to add data to dataset
button2 = Button(left_frame, text="Add Inverted", command=append_data_inverted, width=15, height=2).pack(pady=20) #
button3 = Button(left_frame, text="Next", command=next_image, width=15, height=2).pack()# gen new image

#------------------------------- RIGHT FRAME -----------------------------------
# containg a window that allows us to observe impacts of diff thresholding values
right_frame = Frame(root, width=650, height=400)
right_frame.grid(row=0, column=1, sticky="nsew")

### MY NAME IS DARREN THE COOL DJ NO ONE MESSES AROUND WITH ME 
temp_img = np.array(image.convert('L'))
init_thresh_val, _ = cv2.threshold(temp_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

original_thresholded_img = convert_approach_one(np_image, init_thresh_val, 255) # thresh init image to be displayed by default
original_thresholded_img = rescaled_img(original_thresholded_img, 600)

original_thresholded_img = ImageTk.PhotoImage(Image.fromarray(original_thresholded_img)) # conv back from np.array to photo obj
image_label = Label(right_frame, image=original_thresholded_img)
image_label.pack() # assign init image to label

slider = Scale(right_frame, from_=0, to=255, orient="horizontal", command=update_label)
slider.set(init_thresh_val)
slider.pack(pady=10) # slider to control thresh param 

#-------------------------------- RUN ----------------------------------------- 
root.mainloop()

