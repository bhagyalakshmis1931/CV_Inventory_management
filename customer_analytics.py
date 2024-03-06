import cv2
import datetime   
import imutils
import numpy as np
from nms import non_max_suppression_fast
from centroidtracker import CentroidTracker
import pandas as pd
from collections import defaultdict
protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
# Only enable it if you are using OpenVino environment
# detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)


def main():
    cap = cv2.VideoCapture('test_video.mp4')#0 for webcam 

    fps_start_time = datetime.datetime.now()
    fps = 0
    min=1
    total_frames = 0
    lock=0
    time=0 
    lock1=False
    temp=[]
    tmp=[]
    elapsed_dict=defaultdict(list)
    temp_tup=()
    object_id_list = []
    dtime = dict()
    dwell_time = dict()
    my_dict = {"Count":[],"Time":[],"Elapsed_time":[]}
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
 
        detector.setInput(blob)
        person_detections = detector.forward()
        rects = [] 
        
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        objects = tracker.update(rects)
        #print(type(objects))
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            if objectId not in object_id_list:
                object_id_list.append(objectId)
                now=datetime.datetime.now()
                dtime[objectId] = datetime.datetime.now()
                dwell_time[objectId] = 0
                lock=0 
                tmp.append(0)
                time = now.strftime("%y-%m-%d %H:%M:%S")
                #print(type(objectId))
                my_dict["Count"].append((str(objectId+1)))
                my_dict["Time"].append(str(time))
            else:
                curr_time = datetime.datetime.now()
                old_time = dtime[objectId]
                time_diff = curr_time - old_time
                dtime[objectId] = datetime.datetime.now()
                sec = time_diff.total_seconds()
                dwell_time[objectId] += sec
                elapsed_dict[objectId].append(int(dwell_time[objectId]))
                #my_dict["Elapsed_time"].append(str(dwell_time[objectId]))
                #tmp.append(int(dwell_time[objectId]))
                #print(int(dwell_time[objectId]))
                #print(type(objectId))
                #temp.append(tmp)
                #tmp.pop(0)
                #temp.insert(int(objectId),tmp[-1])
                #if(int(tmp[-1]>10) and lock==0):
                #     print('Anomaly')
                #     lock=1
                #tmp=temp
                #temp_tup=tuple(str(tmp[-1])) 
                #my_dict["Elapsed_time"]=(tmp[-1])                #tmp.clear()
                #if(int(tmp[0])>10 and lock==0):
                #    min+=1
                #    tmp=0
                #    lock=1
                #time = now.strftime("%H:%M:%S")
                #print((int(tmp)))
                #print(int(tmp))
                #if(int(tmp)>20):
                #   lock= not lock

                #if(int(dwell_time[objectId])>20 and min%2==0):
                    #lock=not lock
                   #lock= not lock
                   #print(lock)
                #    min=1
                #    print('Anomaly')
                #if(int(tmp)>20 and lock1==False):
                #   lock11=not lock1  
                #   lock=not lock
               
               # if(int(tmp)>20):
               #    lock= not lock
               # #if(int(dwell_time[objectId]-15)>0):
                #   lock=0
                #   tmp=0
                #if((objectId-len(object_id_list)==0): 
                #my_dict["time"].append([tmp])
                #while(1):
                #   print()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "{}|{}".format(objectId+1, int(dwell_time[objectId]))
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)


        cv2.imshow("Application", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            #print(temp[0])
            #print(my_dict)
            #print((dict(elapsed_dict)))
            #print(elapsed_dict)
            mydict=dict(elapsed_dict)
            #print((mydict[0][-1]))
            tmp_list=[mydict[x][-1] for x in range(len(mydict))]
            print(tmp_list)
            #my_dict["Elapsed_time"].append(max(mydict.values))
            my_dict={"Count":my_dict["Count"],"Time":my_dict["Time"],"Elapsed_time":tmp_list}
            #my_dict = {"Id":[],"Time":[],"Elapsed_time":0}
            print(my_dict)
            #my_dict={"Id":my_dict["Id"],"Time":my_dict["Time"],"Elapsed_time":max(my_dict["Elapsed_time"])}
            #print(my_dict)
            df=pd.DataFrame.from_dict(my_dict)
            #for k,v in dict(elapsed_dict): 
            #     dict[v]=max(dict[v])     
            df.to_csv('çustomer_analytics.csv', index=False)   
            #for l in elapsed_dict.values():
            #     l=max(l)
            #print(my_dict)
            #df=pd.DataFrame.from_dict(elapsed_dict)
            #df.set_index('In_time', inplace=True)
            #print(df)
            #print(df[df["time"]>"16:46:00"])
            break

    cv2.destroyAllWindows()


main() 

