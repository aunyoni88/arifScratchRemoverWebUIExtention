from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import RedirectResponse, FileResponse
from fastapi import File, UploadFile, Form

import gradio as gr
import time
from datetime import datetime, timezone

from pipeline_stable_diffusion_controlnet_inpaint import *
from scratch_detection import ScratchDetection

from arif_install import downloadScratchRemoverModel
# from arifScretchRemover import generate_scratch_mask

from PIL import Image
import cv2
import glob
import shutil
import os
import base64

device = "cuda"


def scratch_remove_api(_: gr.Blocks, app: FastAPI):
    @app.post('/sdapi/ai/v1/scratchRemove/downloadModel')
    async def download_model(
    ):
        #  Source image must be an image
        utc_time = datetime.now(timezone.utc)
        start_time = time.time()

        downloadScratchRemoverModel()

        end_time = time.time()
        server_process_time = end_time - start_time

        return {
            "model_download_time": server_process_time
        }

    @app.post('/sdapi/ai/v1/scratchRemove/generateMask')
    async def generate_mask_image(
            source_image: UploadFile = File()
    ):

        start_time = time.time()

        generate_scratch_mask(source_image)

        end_time = time.time()
        server_process_time = end_time - start_time
        return {
            "server_process_time": server_process_time
        }

    @app.post('/arifTest')
    async def arifTest():
        return {
            "server_process_time": " "
        }

    def generate_scratch_mask(source_image: UploadFile):
        curDir = os.getcwd()
        # input_dir = curDir + "/extensions/arifScratchRemoverWebUIExtention/Arif/"

        fileName = source_image.filename
        filename_without_extention = os.path.splitext(fileName)[0]

        # source_file_location = input_dir + source_image.filename
        # save_file(source_image, source_file_location)
        # input_image = PIL.Image.open(source_file_location).convert('RGB')
        # input_image = PIL.Image.open(img_p).convert('RGB')

        input_path = curDir + "/extensions/arifScratchRemoverWebUIExtention/input_images"
        output_dir = curDir + "/extensions/arifScratchRemoverWebUIExtention/output_masks"

        # Save the input image to a directory
        remove_all_file_in_dir(folder=("%s/*" % input_path))
        source_file_location = input_path + "/" + fileName
        save_file(source_image, source_file_location)

        # remove_all_file_in_dir(folder=("%s/*" % input_path))
        # input_image_path = (f'{input_path}/{fileName}')
        # # input_image_resized = resize_image(input_image, 768)
        # input_image.save(input_image_path)

        scratch_detector = ScratchDetection(input_path, output_dir, input_size="scale_256", gpu=0)
        scratch_detector.run()

        pngExt = '.png'
        mask_image = scratch_detector.get_mask_image((filename_without_extention + pngExt))

        # Resize the mask to match the input image size
        mask_image = mask_image.resize(mask_image.size, Image.BICUBIC)

        # Apply dilation to make the lines bigger
        kernel = np.ones((5, 5), np.uint8)
        mask_image_np = np.array(mask_image)
        mask_image_np_dilated = cv2.dilate(mask_image_np, kernel, iterations=2)
        mask_image_dilated = Image.fromarray(mask_image_np_dilated)

        # window_name = 'Output Image'
        # cv2.imshow(window_name, mask_image_dilated)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # return mask_image_dilated

    def save_file(file: UploadFile, path: str):
        with open(path, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)

    def resize_image(image, target_size):
        width, height = image.size
        aspect_ratio = float(width) / float(height)
        if width > height:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_width = int(target_size * aspect_ratio)
            new_height = target_size
        return image.resize((new_width, new_height), Image.BICUBIC)

    def remove_all_file_in_dir(folder):
        # '/YOUR/PATH/*'
        files = glob.glob(folder)
        for f in files:
            os.remove(f)


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(scratch_remove_api)

except:
    pass
