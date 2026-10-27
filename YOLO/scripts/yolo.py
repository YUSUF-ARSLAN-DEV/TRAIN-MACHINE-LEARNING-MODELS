from ultralytics import YOLO 
from pathlib import Path
from PIL import ImageDraw
import numpy as np
import shutil
import onnxruntime as ort
from PIL import Image
from common.seed import set_seed
from config import (SEED , DATA_YAML , INITIAL_WEIGHTS , RUN_NAME , YOLO_DIR , BEST_PT , BEST_ONNX ,
                    TEST_IMAGES_DIR , OUTPUT_DIR , BATCH_SIZE , WORKERS , EPOCHS , IMGSZ ,
                    CONF_THRESHOLD , CLASS_NAMES)
set_seed(SEED)
def CxCywh_to_xyxy(boxes):
    Cx  = boxes [:,0] 
    Cy = boxes[:,1]
    w = boxes[:,2]
    h = boxes[:,3]
    # calculating top left coordinates 
    x1 = Cx - (w/2) 
    y1  = Cy - (h/2) 

    # calculating the value for the bottom rigfht 
    x2 = Cx  + (w/2) 
    y2 = Cy +  (h/2) 
    return np.stack([x1,y1,x2,y2],axis=1 )  # this reutrns an array of N,4 Dimeniosn

def iou(keeper , boxes) : 
    ix1 = np.maximum(keeper[0] , boxes[:,0] ) 
    iy1 = np.maximum(keeper[1] , boxes[: , 1]) # getting the y1 value of the top left corner
# creating the path to read the unfinte tuned weights 
'''
model_yolo = YOLO(str(INITIAL_WEIGHTS))
model_yolo.train(
    data = str(DATA_YAML) ,
    batch = BATCH_SIZE ,
    workers= WORKERS ,
    cache=False ,
    epochs = EPOCHS ,
    imgsz=IMGSZ,
    name=RUN_NAME ,
    project=str(YOLO_DIR) ,
    exist_ok = True ,
    seed=SEED
)
''' # The training loop
'''
model_yolo = YOLO(str(BEST_PT))
model_yolo.export(format="onnx")
'''
#------------ Running the exported onnx mode
if  __name__ == '__main__' :
    session = ort.InferenceSession(str(BEST_ONNX))
    OUTPUT_DIR.mkdir(exist_ok=True) # save() does not create folders
    x = 0 # numbers the saved test images
    for img_path in TEST_IMAGES_DIR.glob("*.jpg"):
        img =  Image.open(img_path).resize((IMGSZ,IMGSZ))
        # next is t conver the PIL format image intoo a  numpy array  that has shape of 640,640,3 
        arr = np.array(img).astype(np.float32)/255.0 # we are diving by 255 to normalize the value 
        arr = arr.transpose(2,0,1)  # rearranging the array to have channels first then hegiht then width 
        arr = np.expand_dims(arr,0) # adding a batch dimensions so that we can process images in batches 
        # running the inference /
        outputs = session.run(None , {session.get_inputs()[0].name:arr}) 
        pred = outputs[0][0] # the predictions were orignall in this formaat 12,8400 so 12 columns so basically 
        pred = pred.T 
        boxes = pred[: , :4]
        scores = pred[:,4:]
    

        # among the 8400 rows finding the highest class score and which class is it 
        class_ids = np.argmax(scores,axis=1) 
        confidences = np.max(scores,axis=1) 
        to_be_kept = confidences > CONF_THRESHOLD
        class_ids = class_ids[to_be_kept] 
        confidences = confidences[to_be_kept]
        boxes = boxes[to_be_kept] # defining the boxes for a single iamge 

        # the loop that draws all th eboxe s
        xyxy = CxCywh_to_xyxy(boxes) # converting the data for a single image to draw the boxes # processing the 4 values that the model Predicted into coordinates to draw the boxes
        # creating the drawing object that will allow us to draw on the image that we read
        draw = ImageDraw.Draw(img)
        for i in range(len(xyxy)) :
            x1,y1,x2,y2 = xyxy[i]
            label = f"{CLASS_NAMES[class_ids[i]]} {confidences[i]:.2f}"
            draw.rectangle([x1,y1,x2,y2] , outline="black" , width=4 )
            draw.text((x1,y1-10) , label , fill="yellow")

        img.save(OUTPUT_DIR / f"{x+1}.jpg") # you only save once for every image
        x += 1
        
        