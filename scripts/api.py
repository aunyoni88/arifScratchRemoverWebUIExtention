from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import RedirectResponse, FileResponse
from fastapi import File, UploadFile, Form
from typing import List, Optional, Tuple
import gradio as gr
from modules.api import api
import os
import shutil
import time
from datetime import datetime, timezone
import base64
import cv2

from scratchDetection.arifScretchRemover import generate_scratch_mask
from scratchDetection.arif_install import dodnloadScratchRemoverModel


def scratch_remove_api(_: gr.Blocks, app: FastAPI):
    @app.post('/sdapi/ai/v1/scratchRemove/downloadModel')
    async def download_model(
    ):
        #  Source image must be an image
        utc_time = datetime.now(timezone.utc)
        start_time = time.time()

        dodnloadScratchRemoverModel()

        end_time = time.time()
        server_process_time = end_time - start_time

        return {
            "model_download_time": server_process_time
        }

    @app.post('/sdapi/ai/v1/scratchRemove/generateMask')
    async def generate_mask_image(
        # quality: str = Form("80", title = 'output image quality'),
        # source_image: UploadFile = File(),
        # target_image: UploadFile = File(None),
        # target_image_name: str = Form("", title='target image name including extension')
    ):

        start_time = time.time()

        
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





try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(scratch_remove_api)
    
except:
    pass
