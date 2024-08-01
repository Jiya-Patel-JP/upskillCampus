import tkinter as tk
from tkinter import END, filedialog, messagebox, simpledialog
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Convolution2D
from keras.models import Sequential
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Define the path to the directory containing the "crop" and "weed" subdirectories
dataset_path = "dataset"

# Initialize empty lists to store image data (X) and labels (Y)
X = []
Y = []

# Load the class labels from the "classes.txt" file
class_file_path = os.path.join(dataset_path, "classes.txt")
with open(class_file_path, "r") as class_file:
    class_labels = class_file.read().splitlines()

# Use label encoding to map class names to numeric labels
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(class_labels)

# Iterate through the subdirectories ("crop" and "weed")
for class_label, class_name in enumerate(["crop", "weed"]):
    class_dir = os.path.join(dataset_path, class_name)
    
    # Iterate through image files in each class subdirectory
    for filename in os.listdir(class_dir):
        if filename.endswith(".jpg"):  # Adjust the file extension as needed
            try:
                # Load and preprocess the image (e.g., resizing, normalization)
                image = Image.open(os.path.join(class_dir, filename))
                image = image.resize((64, 64))  # Resize to a common size
                image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

                # Append the image data to X
                X.append(image)

                # Assign labels from numeric_labels to Y based on the class label
                Y.append(class_label)

            except Exception as e:
                print(f"Error loading image {filename}: {e}")

# Convert the lists of images and labels to NumPy arrays
X = np.array(X)
Y = np.array(Y)

# Save the NumPy arrays to files
np.save('X.npy', X)
np.save('Y.npy', Y)

print(f"Loaded {len(X)} images and labels.")
print("X.npy and Y.npy files created.")

# Create the main window
main = tk.Tk()
main.title("CROP AND WEED DETECTION")
main.geometry("1050x600")  # Adjusted window size

# Define custom colors
bg_color = "#008000"  # green
text_color = "white"  # Changed text color to black
button_bg_color = "lightgray"  # Changed button background color
button_fg_color = "black"  # Changed button text color
button_font = ('Times New Roman', 12)  # Changed font to Times New Roman

# Set the background color for the main window
main.config(bg=bg_color)


# Global variables
filename = ""
classifier = None
svm_sr, cnn_sr = 0, 0
X, Y = None, None
X_train, X_test, y_train, y_test = None, None, None, None
pca = None


# Function to upload the dataset directory
def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir="feature")
    text.delete('1.0', tk.END)
    text.insert(tk.END, filename + " loaded\n")

# Function to split the dataset
def splitDataset():
    global X, Y
    global X_train, X_test, y_train, y_test
    global pca
    text.delete('1.0', END)
    
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    
    # Check the shape of X
    print("Original X shape:", X.shape)
    
    # Assuming you have 4 dimensions (adjust this according to your data)
    # Reshape X to a 2D array
    X = np.reshape(X, (X.shape[0], -1))
    
    pca = PCA(n_components=100)
    X = pca.fit_transform(X)
    
    print("Reshaped X shape:", X.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    text.insert(END, "Total Images Found in dataset : " + str(len(X)) + "\n")
    text.insert(END, "Train split dataset to 80% : " + str(len(X_train)) + "\n")
    text.insert(END, "Test split dataset to 20%  : " + str(len(X_test)) + "\n")
# Function to execute SVM
def executeSVM():
    global classifier, svm_sr
    text.delete('1.0', tk.END)
    cls = svm.SVC()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    svm_sr = accuracy_score(y_test, predict) * 100
    classifier = cls
    text.insert(tk.END, "SVM Survival Rate: " + str(svm_sr) + "\n")

# Function to execute CNN
def executeCNN():
    global cnn_sr
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    Y = to_categorical(Y)
    
    classifier = Sequential()
    classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Convolution2D(32, 3, 3, activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=256, activation='relu'))
    classifier.add(Dense(units=2, activation='softmax'))  # Fixed the "output_dim" parameter here
    print(classifier.summary())
    
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    hist = classifier.fit(X, Y, batch_size=108, epochs=24, shuffle=True, verbose=2)
    hist = hist.history
    acc = hist['acc']
    cnn_sr = acc[23] * 100
    text.insert(tk.END, "CNN Survival Rate: " + str(cnn_sr) + "\n")


# Function to predict Crop or Weed
def predictCroporWeed():
    global classifier
    filename = filedialog.askopenfilename(initialdir="feature")
    
    if filename:
        img = cv2.imread(filename)

        if img is not None:  # Check if the image is loaded successfully
            img = cv2.resize(img, (64, 64))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(64, 64, 3)
            im2arr = im2arr.astype('float32')
            im2arr = im2arr / 255
            test = []
            test.append(im2arr)
            test = np.asarray(test)
            test = np.reshape(test, (test.shape[0], (test.shape[1]*test.shape[2]*test.shape[3])))
            test = pca.transform(test)
            predict = classifier.predict(test)[0]
            msg = "Uploaded image is crop" if predict == 0 else "Uploaded image is weed"
            img = cv2.imread(filename)
            img = cv2.resize(img, (400, 400))
            cv2.putText(img, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(msg, img)
            cv2.waitKey(0)
        else:
            messagebox.showerror("Error", "Failed to load the image.")
    else:
        messagebox.showinfo("Info", "No image selected.")


# Function to display a bar graph
def graph():
    height = [svm_sr, cnn_sr]
    bars = ('SVM Survival Rate', 'CNN Survival Rate')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

# Create and configure widgets with cus
# tom styles
title = tk.Label(main, text='CROP AND WEED DETECTION')
title.config(bg=bg_color, fg=text_color, font=('Helvetica', 20, 'bold'), pady=20)
title.pack()

text = tk.Text(main, height=15, width=150)
scroll = tk.Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.pack()
text.config(font=('Helvetica', 14))

# Create a frame for buttons
button_frame = tk.Frame(main, bg=bg_color)
button_frame.pack(pady=20)

# Buttons
uploadButton = tk.Button(button_frame, text="Upload Weed or Crop Dataset", command=uploadDataset)
uploadButton.config(bg=button_bg_color, fg=button_fg_color, font=button_font, padx=10, pady=5)
uploadButton.grid(row=0, column=0, padx=10)

readButton = tk.Button(button_frame, text="Read & Split Dataset", command=splitDataset)
readButton.config(bg=button_bg_color, fg=button_fg_color, font=button_font, padx=10, pady=5)
readButton.grid(row=0, column=1, padx=10)

svmButton = tk.Button(button_frame, text="Execute SVM Algorithm", command=executeSVM)
svmButton.config(bg=button_bg_color, fg=button_fg_color, font=button_font, padx=10, pady=5)
svmButton.grid(row=0, column=2, padx=10)

cnnButton = tk.Button(button_frame, text="Execute CNN Algorithm", command=executeCNN)
cnnButton.config(bg=button_bg_color, fg=button_fg_color, font=button_font, padx=10, pady=5)
cnnButton.grid(row=0, column=3, padx=10)

predictButton = tk.Button(button_frame, text="Predict Crop or Weed", command=predictCroporWeed)
predictButton.config(bg=button_bg_color, fg=button_fg_color, font=button_font, padx=10, pady=5)
predictButton.grid(row=0, column=4, padx=10)

graphButton = tk.Button(main, text="Display Graph", command=graph)
graphButton.config(bg=button_bg_color, fg=button_fg_color, font=button_font, padx=10, pady=5)
graphButton.pack(fill=tk.X, padx=20, pady=(0, 10))

main.mainloop()