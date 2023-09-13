from functools import partial

from PIL import Image
import numpy as np
import gradio as gr
import torch
import os
import fire

from ldm.util import add_margin

_TITLE = '''SyncDreamer: Generating Multiview-consistent Images from a Single-view Image'''
_DESCRIPTION = '''
<div>
<a style="display:inline-block" href="https://liuyuan-pal.github.io/SyncDreamer/"><img src="https://img.shields.io/badge/SyncDremer-Homepage-blue"></a>
<a style="display:inline-block; margin-left: .5em" href="https://arxiv.org/abs/2309.03453"><img src="https://img.shields.io/badge/2309.03453-f9f7f7?logo=data:image/png;base64,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"></a>
<a style="display:inline-block; margin-left: .5em" href='https://github.com/liuyuan-pal/SyncDreamer'><img src='https://img.shields.io/github/stars/liuyuan-pal/SyncDreamer?style=social' /></a>
</div>
Given a single-view image, SyncDreamer is able to generate multiview-consistent images, which enables direct 3D reconstruction with NeuS or NeRF without SDS loss'''
_USER_GUIDE0 = "Step0: Please upload an image in the block above (or choose an example above). We use alpha values as object masks if given."
_USER_GUIDE1 = "Step1: Please select a crop size using the glider."
_USER_GUIDE2 = "Step2: Please choose a suitable elevation angle and then click the Generate button."


def mask_prediction(mask_predictor, image_in: Image.Image):
    if image_in.mode=='RGBA':
        return image_in
    else:
        raise NotImplementedError

def resize_inputs(image_input, crop_size):
    alpha_np = np.asarray(image_input)[:, :, 3]
    coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
    min_x, min_y = np.min(coords, 0)
    max_x, max_y = np.max(coords, 0)
    ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
    h, w = ref_img_.height, ref_img_.width
    scale = crop_size / max(h, w)
    h_, w_ = int(scale * h), int(scale * w)
    ref_img_ = ref_img_.resize((w_, h_), resample=Image.BICUBIC)
    results = add_margin(ref_img_, size=256)
    return results


def run_demo():
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    models = None # init_model(device, os.path.join(code_dir, ckpt))

    # init sam model
    mask_predictor = None # sam_init(device_idx)

    # with open('instructions_12345.md', 'r') as f:
    #     article = f.read()

    # NOTE: Examples must match inputs
    example_folder = os.path.join(os.path.dirname(__file__), 'hf_demo', 'examples')
    example_fns = os.listdir(example_folder)
    example_fns.sort()
    examples_full = [os.path.join(example_folder, x) for x in example_fns if x.endswith('.png')]

    # Compose demo layout & data flow.
    with gr.Blocks(title=_TITLE, css="hf_demo/style.css") as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('# ' + _TITLE)
            # with gr.Column(scale=0):
            #     gr.DuplicateButton(value='Duplicate Space for private use', elem_id='duplicate-button')
        gr.Markdown(_DESCRIPTION)

        with gr.Row(variant='panel'):
            with gr.Column(scale=1):
                image_block = gr.Image(type='pil', image_mode='RGBA', height=256, label='Input image', tool=None, interactive=True)
                guide_text = gr.Markdown(_USER_GUIDE0, visible=True)
                gr.Examples(
                    examples=examples_full,  # NOTE: elements must match inputs list!
                    inputs=[image_block],
                    outputs=[image_block],
                    cache_examples=False,
                    label='Examples (click one of the images below to start)',
                    examples_per_page=40
                )

            with gr.Column(scale=1):
                sam_block = gr.Image(type='pil', image_mode='RGBA', label="SAM output", height=256, interactive=False)
                crop_size_slider = gr.Slider(120, 240, 200, step=10, label='Crop size', interactive=True)

            with gr.Column(scale=1):
                input_block = gr.Image(type='pil', image_mode='RGB', label="Input to SyncDreamer", height=256, interactive=False)
                elevation_slider = gr.Slider(-10, 40, 30, step=5, label='Elevation angle', interactive=True)
                run_btn = gr.Button('Run Generation', variant='primary', interactive=False)

        update_guide = lambda GUIDE_TEXT: gr.update(value=GUIDE_TEXT)
        image_block.change(fn=partial(mask_prediction, mask_predictor), inputs=[image_block], outputs=[sam_block], queue=False)\
                   .success(fn=partial(update_guide, _USER_GUIDE1), outputs=[guide_text], queue=False)

        crop_size_slider.change(fn=resize_inputs, inputs=[sam_block, crop_size_slider], outputs=[input_block], queue=False)\
                        .success(fn=partial(update_guide, _USER_GUIDE2), outputs=[guide_text], queue=False)

        run_btn.click

    demo.queue().launch(share=False, max_threads=80)  # auth=("admin", os.environ['PASSWD'])

if __name__=="__main__":
    fire.Fire(run_demo)