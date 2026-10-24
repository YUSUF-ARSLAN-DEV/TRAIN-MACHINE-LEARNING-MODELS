from ultralytics import YOLO 
from pathlib import Path
from PIL import ImageDraw
import numpy as np
import shutil
import onnxruntime as ort
from PIL import Image
names = ["Bottle", "Box", "Tin can", "Snack", "Drink", "Milk", "Personal care", "Cosmetics"] 
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
    iy1 = np.maximum(keeper[0] , boxes[: , 1]) # getting the y1 value of the top left corner
# creating the path to read the unfinte tuned weights 
'''
yolo_dir = Path(__file__).resolve().parent # just getting the parent so that we can move to that director 
model_yolo = YOLO(str(yolo_dir/"yolov8n.pt"))
model_yolo.train( 
    data = "data.yaml" , 
    batch = 48 ,
    workers= 2 , 
    cache=False , 
    epochs = 20 , 
    imgsz=640, 
    name="openimages_v8n_e20" , 
    project=str(yolo_dir) , 
    exist_ok = True , 
) 
''' # The training loop 
'''
model_yolo = YOLO("../openimages_v8n_e20/weights/best.pt")
model_yolo.export(format="onnx")
'''
#------------ Running the exported onnx mode
if  __name__ == '__main__' : 
    directory_with_test_images="dataset/images/test" 
    session = ort.InferenceSession("../openimages_v8n_e20/weights/best.onnx")
    Path("../TEST_RUN_1").mkdir(exist_ok=True) # save() does not create folders
    x = 0 # numbers the saved test images
    for img_path in Path("../dataset/images/test").glob("*.jpg"):
        img =  Image.open(img_path).resize((640,640))
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
        to_be_kept = confidences > 0.25 
        class_ids = class_ids[to_be_kept] 
        confidences = confidences[to_be_kept]
        boxes = boxes[to_be_kept] # defining the boxes for a single iamge 

        # the loop that draws all th eboxe s
        xyxy = CxCywh_to_xyxy(boxes) # converting the data for a single image to draw the boxes # processing the 4 values that the model Predicted into coordinates to draw the boxes
        # creating the drawing object that will allow us to draw on the image that we read
        draw = ImageDraw.Draw(img)
        for i in range(len(xyxy)) :
            x1,y1,x2,y2 = xyxy[i]
            label = f"{names[class_ids[i]]} {confidences[i]:.2f}"
            draw.rectangle([x1,y1,x2,y2] , outline="black" , width=4 )
            draw.text((x1,y1-10) , label , fill="yellow")

        img.save(f"../TEST_RUN_1/{x+1}.jpg") # you only save once for every image
        x += 1
        
        