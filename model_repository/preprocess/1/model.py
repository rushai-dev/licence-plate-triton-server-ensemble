import numpy as np
import sys
import json
import io
import triton_python_backend_utils as pb_utils

from PIL import Image
import torchvision.transforms as transforms
import os


class TritonPythonModel:

    def initialize(self, args):

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT_0")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

    def execute(self, requests):

        output0_dtype = self.output0_dtype

        responses = []

        for request in requests:

            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            loader = transforms.Compose([
                transforms.Resize([640, 640]),
                transforms.ToTensor()
            ])

            def image_loader(image_name):
                image = loader(image_name)
                image = image.unsqueeze(0)
                return image

            img = in_0.as_numpy()

            image = Image.open(io.BytesIO(img.tobytes()))
            img_out = image_loader(image)
            img_out = np.array(img_out)

            out_tensor_0 = pb_utils.Tensor("OUTPUT_0",
                                           img_out.astype(output0_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print('Cleaning up...')
