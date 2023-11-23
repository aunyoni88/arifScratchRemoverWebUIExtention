from fastapi import FastAPI
import gradio as gr
import time
from datetime import datetime, timezone

from arif_install import downloadScratchRemoverModel


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
