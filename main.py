# pip install opencv-python
# pip install tensorflow
# pip install pillow
# pip install playsound
# pip install gtts

# Libraries Importing
import cv2 # Computer Vision Library
import numpy as np # Numerical Python Library
import tensorflow.keras # tensorflow Library
from PIL import Image, ImageOps # Image Library 
import time # time library
import os # os library

from gtts import gTTS # google text to speech
import playsound # sound play

################## LIBRARIES SECTION################
# CONVERT, SAVE, PLAY, REMOVE
# Text to Speech, Play Audio 
def playaudio(t): # text - convert
 language='en' # language select
 m=gTTS(text=t,lang=language,slow=False) # text convert
 m.save('out.mp3') # saving the audio file
 time.sleep(1) # 1 second delay
 # pyaudio 
 playsound.playsound('out.mp3') # play
 os.remove('out.mp3') # remove

####################### TTS, AUDIO PLAY##############

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('model.h5')

g1Flag=0
g2Flag=0
g3Flag=0
g4Flag=0
g5Flag=0
g6Flag=0

##################### MODEL LOADING ##############

def sign_language_detection(a): # abc.jpg

 global g1Flag, g2Flag, g3Flag, g4Flag, g5Flag, g6Flag, g7Flag
 # Create the array of the right shape to feed into the keras model
 # The 'length' or number of images you can put into the array is
 # determined by the first position in the shape tuple, in this case 1.
 data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

 # Replace this with the path to your image
 image = Image.open(a) # opening the  image

 #resize the image to a 224x224 with the same strategy as in TM2:
 #resizing the image to be at least 224x224 and then cropping from the center
 size = (224, 224)
 image = ImageOps.fit(image, size, Image.ANTIALIAS)

 #turn the image into a numpy array
 image_array = np.asarray(image)

 # display the resized image
 #image.show()

 # ########################### NORMALISATION #########################

 # Normalize the image 
 normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

 # Load the image into the array
 data[0] = normalized_image_array

 ########################### IMAGE PRE-PROCESSING STAGE ###############

 # run the inference
 prediction = model.predict(data)
 #print(prediction)

 ########################### PREDICTION ############################

 # Decision Making Model Block
 prediction=list(prediction[0])
 max_prediction=max(prediction)
 index_max=prediction.index(max_prediction)
  
 if(index_max==0):
  g1Flag+=1
 elif(index_max==1):
  g2Flag+=1
 elif(index_max==2):
  g3Flag+=1
 elif(index_max==3):
  g4Flag+=1
 elif(index_max==4):
  g5Flag+=1
 elif(index_max==5):
  g6Flag+=1
 

 if(g1Flag==10):
  print('Gesture1 Found')
  g1Flag=0
 elif(g2Flag==10):
  print('Gesture2 Found')
  playaudio('I like what you are talking about')
  time.sleep(2)
  g2Flag=0
 elif(g3Flag==10):
  print('Gesture3 Found')
  playaudio('I dislike on your statements')
  g3Flag=0
 elif(g4Flag==10):
  print('Gesture4 Found')
  playaudio('Hi, Hello')
  g4Flag=0
 elif(g5Flag==10):
  print('Gesture5 Found')
  playaudio('Super Iam excited')
  g5Flag=0
 elif(g6Flag==10):
  print('Gesture6 Found')
  playaudio('I have a doubt')
  g6Flag=0

############################## DECISION MAKING BLOCK ####################

# Camera Block
video = cv2.VideoCapture(0) # Open Camera - 0
video.set(cv2.CAP_PROP_FRAME_WIDTH,240) # W - 240
video.set(cv2.CAP_PROP_FRAME_HEIGHT,240) # H - 240

# Image Acquisition Block
while True:
 res, frame = video.read() # reading frames from the camera
 if res==1: # frames are available
  #Convert the captured frame into RGB
  cv2.imwrite('abc.jpg', frame) # Saving the Image

  # Deep Learning Neural Network Block
  sign_language_detection('abc.jpg')
  cv2.imshow("Capturing", frame)
  key=cv2.waitKey(1)
  if key == ord('q'):
   break
video.release()
cv2.destroyAllWindows()




