import cv2
import glob
import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#loads a set of training images from a particular path, assigning the supplied label to each image 
def load_training_set(path_to_training_images, label_to_assign):
    X_train = []    #training examples (images)
    y_train = []    #labels: this is binary classification so 0 or 1
    #enumerate image file path list, loading each image, converting to RGB, and adding it to the X_train list 
    for cur_image_file_path in glob.iglob(path_to_training_images, recursive=True):
        #load current image
        bgr_image = cv2.imread(cur_image_file_path)
        #convert current image from BGR (opencv standard) to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        #append loaded image to X_train list
        X_train.append(rgb_image)
        #append supplied label to y_train list
        y_train.append(label_to_assign)
    #return tuple of training images and labels
    return (np.array(X_train), np.array(y_train))

####################################
## LOAD, STACK, SHUFFLE DATA SETS ##
####################################

print()
print("Building training data set from provided vehicle and non-vehicle images...")
print()

#vehicles (the '**' pattern designates recursion)
X_train_vehicles, y_train_vehicles = load_training_set("data_sets/vehicles/project_vehicles/**/*.png", label_to_assign=1)
#non-vehicles (the '**' pattern designates recursion)
X_train_non_vehicles, y_train_non_vehicles = load_training_set("data_sets/non_vehicles/project_non_vehicles/**/*.png", label_to_assign=0)

print("Num vehicle images:", X_train_vehicles.shape)
print("Num vehicle labels:", y_train_vehicles.shape)
print("Num non-vehicle images:", X_train_non_vehicles.shape)
print("Num non-vehicle labels:", y_train_non_vehicles.shape)

#stack the two data sets into a single training set
X_train = np.vstack((X_train_vehicles, X_train_non_vehicles)) #vertical stack
y_train = np.hstack((y_train_vehicles, y_train_non_vehicles)) #horizontal stack

#shuffle the training set before carving out the test set
X_train, y_train = shuffle(X_train, y_train)

########################
## CARVE OUT TEST SET ##
########################

print()
print("Carving out test data set...")
print()

#carve out a portion of the training set to use for model validation
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)

print("Num X_train images:", X_train.shape)
print("Num y_train labels:", y_train.shape)
print("Num X_test images:", X_test.shape)
print("Num y_test labels:", y_test.shape)

#################################
## PICKLE TRAINING & TEST SETS ##
#################################

print()
print("Pickling training and test data sets...")
print()

#pickled file names
training_set_file = "pickled_objects/training_set.p"
test_set_file = "pickled_objects/test_set.p"

#embed data in dictionary
training_set_dict = {"features": X_train, "labels": y_train}
test_set_dict = {"features": X_test, "labels": y_test}

#pickle training data
with open(training_set_file, mode="wb") as f:
    pickle.dump(training_set_dict, f)
#pickle test data
with open(test_set_file, mode="wb") as f:
    pickle.dump(test_set_dict, f)
    
print("Done.")
print()