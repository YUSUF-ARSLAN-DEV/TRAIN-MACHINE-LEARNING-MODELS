from ultralytics import YOLO 

if __name__ == '__main__' : 
    modelyolo = YOLO(r"C:\Users\aonli\Desktop\INTERNSHIP PROJECTS\Independent Learning\TRAIN MACHINE LEARNING MODELS\runs\detect\train-6\weights\best.pt")
    #modelyolo.train(data='data.yaml',epochs=50,imgsz=640) 
    results = modelyolo(r"C:\Users\aonli\Desktop\INTERNSHIP PROJECTS\Independent Learning\TRAIN MACHINE LEARNING MODELS\YOLO\test\images\IMG_20220428_120046_jpg.rf.df61b58273986c6bbe6452e356b4bfd9.jpg")
    results[0].show() 
    results[0].save()