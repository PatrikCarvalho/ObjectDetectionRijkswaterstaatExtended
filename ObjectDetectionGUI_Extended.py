# Import Pckackeges
import tkinter as tk
from PIL import Image, ImageTk

import numpy as np

import os

import tensorflow as tf

from utils import label_map_util

from utils import visualization_utils as vis_util

import cv2

import six.moves.urllib as urllib

import tarfile



#### Define start function
def start_visual_detection():
    # What model to download.
    ModelName_Dictionary = {
        "ssd_mobilenet_v2_coco (22)": "ssd_mobilenet_v2_coco_2018_03_29",
        "ssd_inception_v2_coco (24)": "ssd_inception_v2_coco_2018_01_28",
        "faster_rcnn_inception_v2_coco (28)": "faster_rcnn_inception_v2_coco_2018_01_28",
        "faster_rcnn_resnet101_coco (32)": "faster_rcnn_resnet101_coco_2018_01_28",
        "faster_rcnn_inception_resnet_v2_atrous_coco (37)": "faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"
        }
    MODEL_NAME = ModelName_Dictionary[Model_Type_Var.get()]
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    
    NUM_CLASSES = 90

    # ## Download Model
    if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
        print ('Downloading the model')
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                  tar_file.extract(file, os.getcwd())
        print ('Download complete')
    else:
    	  print ('Model already exists')
    
    # ## Load a (frozen) Tensorflow model into memory.
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    

    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    #intializing the web camera device
    
    
    cap = cv2.VideoCapture(0)
    
    # Running the tensorflow session
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            ret = True
            Stop_detection_var.set(False)
            while(ret):
                ret,image_np = cap.read()
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                #      plt.figure(figsize=IMAGE_SIZE)
                #      plt.imshow(image_np)
                cv2.imshow('image',cv2.resize(image_np,(900,800)))

                # Conditions for closing object detection windows and stop detection
                # Condition 1
                if cv2.waitKey(25) & 0xFF == ord('q') or Stop_detection_var.get() == '1':
                    cv2.destroyAllWindows()
                    cap.release()
                    break

                # Calculating the amount of person and objects in detection
                amount_person = 0
                amount_objects = num_detections[0].astype(np.int32)
                min_score_thresh = 0.5
                for i in range(len(np.squeeze(classes)) - 1):
                    if np.squeeze(classes)[i] == 1 and np.squeeze(scores)[i] >= min_score_thresh:
                        amount_person = amount_person + 1
                        amount_objects = amount_objects - 1

                # Set the stringvariables and update window
                Detections_person_lbl.set(str(amount_person))
                Detections_lbl.set(str(num_detections[0].astype(np.int32)))
                Detections_object_lbl.set(str(amount_objects))
                window.update()




# Condition 2 for stopping  object detection
#### Define stop function
def stop_visual_detection():
    Stop_detection_var.set(True)







## main:

#### Options for different model types

# Define pretrained models the users can choose from
MODEL_TYPE_Options = [
"ssd_mobilenet_v2_coco (22)",       # Model_Name = "ssd_mobilenet_v2_coco_2018_03_29"
"ssd_inception_v2_coco (24)",       # Model_Name = "ssd_inception_v2_coco_2018_01_28"
"faster_rcnn_inception_v2_coco (28)",  # Model_Name = "faster_rcnn_inception_v2_coco_2018_01_28"
"faster_rcnn_resnet101_coco (32)",    # Model_Name = "faster_rcnn_resnet101_coco_2018_01_28"
"faster_rcnn_inception_resnet_v2_atrous_coco (37)"  # Model_Name = "faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"
] #etc

### Start making GUI
window = tk.Tk()
window.title("Object Detectie GUI")
#window.geometry("1500x800")
window.configure(background="white")



### Introduce RWS Logo
image = Image.open("RWSLogo2.png")
photo1 = ImageTk.PhotoImage(image)
tk.Label(window, image=photo1, bg="white").grid(row=0, column=2, sticky=tk.E)


#### Add a START buton
tk.Button(window, text="START", width=15, command=start_visual_detection).grid(row=1, column=0, sticky=tk.W)
#### Create a Label for START button
tk.Label(window, text="Start detectie:", bg="white", fg="black", font="none 12 bold").grid(row=0, column=0, sticky=tk.SW)


#### Add a STOP button
Stop_detection_var = tk.StringVar()
tk.Button(window, text="STOP", width=14, command=stop_visual_detection).grid(row=1, column=1, sticky=tk.W)
#### Create a Label for START button
tk.Label(window, text="Stop detectie:", bg="white", fg="black", font="none 12 bold").grid(row=0, column=1, sticky=tk.SW)




#### Create a dynamic Label for showing number of detections
Detections_lbl = tk.StringVar()
tk.Label(window, textvariable=Detections_lbl, bg="white", fg="black", font="none 12 bold").grid(row=1, column=4, sticky=tk.W)
#### Create a Label for the number of detections
tk.Label(window, text="Aantal Detecties:", bg="white", fg="black", font="none 12 bold").grid(row=1, column=3, sticky=tk.W)


#### Create a dynamic Label for showing number of persons detected
Detections_person_lbl = tk.StringVar()
tk.Label(window, textvariable=Detections_person_lbl, bg="white", fg="black", font="none 12 bold").grid(row=2, column=4, sticky=tk.W)
#### Create a Label for the number of detections
tk.Label(window, text="Personen:", bg="white", fg="black", font="none 12 bold").grid(row=2, column=3, sticky=tk.W)



#### Create a dynamic Label for showing number of persons detected
Detections_object_lbl = tk.StringVar()
tk.Label(window, textvariable=Detections_object_lbl, bg="white", fg="black", font="none 12 bold").grid(row=3, column=4, sticky=tk.W)
#### Create a Label for the number of detections
tk.Label(window, text="Objecten:", bg="white", fg="black", font="none 12 bold").grid(row=3, column=3, sticky=tk.W)



#### Create a dropdown menu for different modeltypes
Model_Type_Var = tk.StringVar()
Model_Type_Var.set(MODEL_TYPE_Options[0]) # default value
tk.OptionMenu(window, Model_Type_Var, *MODEL_TYPE_Options).grid(row=3, column=0, sticky=tk.W)
tk.Label(window, text="Kies Model:", bg="white", fg="black", font="none 12 bold").grid(row=2, column=0, sticky=tk.SW)




#### run the main loop
window.mainloop()




















