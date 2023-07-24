# import the opencv library
import cv2
import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model("C:/Users/asus/Downloads/converted_keras (1)/keras_model.h5")
  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    img=cv2.resize(frame,(224,224))
    img1=np.array(img,dtype=np.float32)
    img1=np.expand_dims(img1,axis=0)
    nimg=img1/255.0
    prediction=model.predict(nimg)
    rock=int(prediction[0][0]*100)
    paper=int(prediction[0][1]*100)
    scissor=int(prediction[0][2]*100)
    # predict_class=np.argmax(prediction,axis=1)
    # print(predict_class)
    print(f"Rock: {rock} %, Paper: {paper} %, Scissor: {scissor} %")
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()