from ultralytics import YOLO 

if __name__ == '__main__' : 
    modelyolo = YOLO(r"C:\Users\aonli\Desktop\INTERNSHIP PROJECTS\Independent Learning\TRAIN MACHINE LEARNING MODELS\runs\detect\train-6\weights\best.pt")
    #modelyolo.train(data='data.yaml',epochs=50,imgsz=640) 
    results = modelyolo("EXTERNAL_TEST_SAMPLES/lipton.png") 
    results[0].show() 
    results[0].save()