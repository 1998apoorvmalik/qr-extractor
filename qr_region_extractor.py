import os
import time

import cv2 as cv
import numpy as np


class QRRegionExtractor:
    def __init__(self, model_dir="model"):
        """Deep Learning based image segmentation model used to locate QR codes in an image.

        Args:
            model_dir (str, optional): Path to directory where segmentation model is saved. Defaults to "model".
        """
        self.model_dir = model_dir

        self.classes = open(os.path.join(model_dir, "qrcode.names")).read().strip().split("\n")

        # yolo segmentation network
        self.net = cv.dnn.readNetFromDarknet(
            os.path.join(model_dir, "qrcode-yolov3-tiny.cfg"),
            os.path.join(model_dir, "qrcode-yolov3-tiny.weights"),
        )
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

        # dnn layer name
        self.layer_names = self.net.getLayerNames()
        self.layer_names = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def preprocess_image(self, frame, sensitivity=70):
        """Preprocess the input image

        Args:
            frame (numpy.ndarray): Input image frame to preprocess.
            sensitivity (int, optional): White region threshold sensitivity. Defaults to 70.

        Returns:
            preprocessed_image (numpy.ndarray): preprocessed image frame.
        """
        # convert original loaded image format: from RGB to HSV
        hsv_image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # lower white & upper white thresholds
        lower_white = np.array([0, 0, 255 - sensitivity])
        upper_white = np.array([255, sensitivity, 255])

        # threshold the HSV image to get only white colors
        mask = cv.inRange(hsv_image, lower_white, upper_white)

        # bitwise-AND mask and original image
        preprocessed_image = cv.bitwise_and(frame, frame, mask=mask)

        return preprocessed_image

    def forward(self, frame, debug_mode=False):
        """Forward the input frame through the segmentation network, then return the output.

        Args:
            frame (numpy.ndarray): Input frame to forward through the network.
            debug_mode (bool, optional): Whether to display debug information. Defaults to False.

        Returns:
            [type]: [description]
        """
        # convert frame to blob
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)

        self.net.setInput(blob)

        # compute the network output
        start_time = time.monotonic()
        network_outs = self.net.forward(self.layer_names)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        # print forward time if debug mode is set to true
        if debug_mode:
            print("[INFO] forward done in %.1fms" % (elapsed_ms))

        return network_outs

    def get_qr_region(self, image, sensitivity=70, confidence_threshold=0.25, debug_mode=False):
        """Extract QR regions from a given image.

        Args:
            image_path (numpy.ndarray): Input image from which QR region is to be extracted.
            sensitivity (int, optional): White region threshold sensitivity. Defaults to 70.
            confidence_threshold (float, optional): Minimum prediction confidence to consider a detected region as
                                                    QR region. Defaults to 0.25.
            debug_mode (bool, optional): Whether to display debug information and images. Defaults to False.

        Returns:
            boxes (list): List of bounding boxes (left, top, width, height) of all detected QR regions.
        """

        # preprocess the loaded image
        frame = self.preprocess_image(image, sensitivity)

        # forward the frame through the segmentation network
        network_outs = self.forward(frame, debug_mode)

        frameHeight, frameWidth = frame.shape[:2]

        classIds = []
        confidences = []
        boxes = []

        for out in network_outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confidence_threshold:
                    x, y, width, height = detection[:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
                    left = int(x - width / 2)
                    top = int(y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, int(width), int(height)])

        if debug_mode:
            indices = cv.dnn.NMSBoxes(
                boxes,
                confidences,
                confidence_threshold,
                confidence_threshold - 0.1,
            )
            for i in indices:
                i = i[0]
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]

                # draw bounding box for objects
                cv.rectangle(
                    frame,
                    (left, top),
                    (left + width, top + height),
                    (0, 0, 255),
                )

                # draw class name and confidence
                label = "%s:%.2f" % (self.classes[classIds[i]], confidences[i])
                cv.putText(
                    frame,
                    label,
                    (left, top - 15),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                )

            cv.imshow("Debug Image", np.concatenate((image, frame), axis=1))
            cv.waitKey(0)
            cv.destroyAllWindows()

        return boxes
