# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.1,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.1,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = "/home/adithya/Desktop/Yolo/yolo-coco/cone.names"
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = "/home/adithya/Desktop/Yolo/yolo-coco/yolo_cone.weights"
configPath = "/home/adithya/Desktop/Yolo/yolo-coco/yolo_cone.cfg"

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our input image and grab its spatial dimensions
cap = cv2.VideoCapture("/home/adithya/Desktop/Yolo/vids/3.mp4")
out = cv2.VideoWriter('/home/adithya/Desktop/Yolo/output8.avi', cv2.VideoWriter_fourcc(*'MJPG') ,20.0, (1920,1080))
while True:
	# load our input image and grab its spatial dimensions
	
	ret,image = cap.read()
	if(ret):
		(H, W) = image.shape[:2]

		# determine only the *output* layer names that we need from YOLO
		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

		# construct a blob from the input image and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes and
		# associated probabilities
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()

		# show timing information on YOLO
		print("[INFO] YOLO took {:.6f} seconds".format(end - start))

		# initialize our lists of detected bounding boxes, confidences, and
		# class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability) of
				# the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > args["confidence"]:
					# scale the bounding box coordinates back relative to the
					# size of the image, keeping in mind that YOLO actually
					# returns the center (x, y)-coordinates of the bounding
					# box followed by the boxes' width and height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top and
					# and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates, confidences,
					# and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		# apply non-maxima suppression to suppress weak, overlapping bounding
		# boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
			args["threshold"])

		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				x1=x
				x2=x+w
				y1=y
				y2=y+w
				c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
				colr = np.mean(image[(c1[1]+c2[1])//2:c2[1], int(0.5 * (c1[0] + c2[0])), :],axis=0)
				b,g,r=colr/np.sum(colr)
				if b>0.4:
					label='blue'
				elif g > 0.5:
					label = 'green'
				elif r > 0.5:
					label = 'orange'
				elif g>0.3 and r>0.3:
					label = 'yellow'
				else:
					label = 'idk'
				print(label)
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
				
				cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
					0.5, color, 2)


		# show the output image
		cv2.imshow("Image", image)
		out.write(image)

		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

	else:
		break
# clear everything once finished
print("done")
cap.release()
out.release()
cv2.destroyAllWindows()