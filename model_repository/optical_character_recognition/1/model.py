import numpy as np
import sys
import json
import io
import easyocr

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

from PIL import Image
import torchvision.transforms as transforms
import os
import cv2


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        self.conf_thresold=0.25

        self.input_width=640
        self.input_height=640
        
        self.image_width=640
        self.image_height=640

        self.reader = easyocr.Reader(['th'] ,gpu=True)

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT_0")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

    def preprocess(self, image, alpha = 2.0, beta = 90):
        # Convert color images to black and white images.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        
        # Adjust the sharpness of the image.
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
        
        # Adjust the brightness and intensity of the image.
        adjusted = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
            
        return adjusted

    def compute_iou(self, box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou

    def nms(self, boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]

            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes

    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")

            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT_1")

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            loader = transforms.Compose([
                transforms.Resize([640, 640]),
                transforms.ToTensor()
            ])

            def image_loader(image_name):
                image = loader(image_name)
                #expand the dimension to nchw
                image = image.unsqueeze(0)
                return image

            img = in_1.as_numpy()
            
            predictions = np.squeeze(in_0.as_numpy()).T

            # Filter out object confidence scores below threshold
            scores = np.max(predictions[:, 4:], axis=1)
            predictions = predictions[scores > self.conf_thresold, :]
            scores = scores[scores > self.conf_thresold]  

            class_ids = np.argmax(predictions[:, 4:], axis=1)

            # Get bounding boxes for each object
            boxes = predictions[:, :4]

            #rescale box
            input_shape = np.array([self.input_width, self.input_height, self.image_width, self.image_height])
            boxes = np.divide(boxes, input_shape, dtype=np.float32)
            boxes *= np.array([image_width, image_height, image_width, image_height])
            boxes = boxes.astype(np.int32)

            indices = self.nms(boxes, scores, 0.3)

            x1, y1, x2, y2 = self.xywh2xyxy(boxes[indices])[0]

            image = Image.open(io.BytesIO(img.tobytes()))

            image = preprocess(image=np.array(image)[y1:y2, x1:x2])

            results = self.reader.readtext(image, detail=0)

            str_out = np.array([results[0], results[1], x1, y1, x2, y2])

            out_tensor_0 = pb_utils.Tensor("OUTPUT_0",
                                           str_out.astype(output0_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
