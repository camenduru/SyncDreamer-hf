import torch
import numpy as np
from PIL import Image
from skimage.io import imsave
from sam_utils import sam_out_nosave, sam_init

class BackgroundRemoval:
    def __init__(self, device='cuda'):
        from carvekit.api.high import HiInterface
        self.interface = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device=device,
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=231,
            trimap_dilation=30,
            trimap_erosion_iters=5,
            fp16=True,
        )

    @torch.no_grad()
    def __call__(self, image):
        # image: [H, W, 3] array in [0, 255].
        # image = Image.fromarray(image)
        image = self.interface([image])[0]
        # image = np.array(image)
        return image

raw_im = Image.open('hf_demo/examples/flower.png')
predictor = sam_init()

raw_im.thumbnail([512, 512], Image.Resampling.LANCZOS)
width, height = raw_im.size
image_nobg = BackgroundRemoval()(raw_im.convert('RGB'))
arr = np.asarray(image_nobg)[:, :, -1]
x_nonzero = np.nonzero(arr.sum(axis=0))
y_nonzero = np.nonzero(arr.sum(axis=1))
x_min = int(x_nonzero[0].min())
y_min = int(y_nonzero[0].min())
x_max = int(x_nonzero[0].max())
y_max = int(y_nonzero[0].max())
image_nobg.save('./nobg.png')

image_nobg.thumbnail([512, 512], Image.Resampling.LANCZOS)
image_sam = sam_out_nosave(predictor, image_nobg.convert("RGB"), (x_min, y_min, x_max, y_max))

imsave('./mask.png', np.asarray(image_sam)[:,:,3])
image_sam = np.asarray(image_sam, np.float32) / 255
out_mask = image_sam[:, :, 3:]
out_rgb = image_sam[:, :, :3] * out_mask + 1 - out_mask
out_img = (np.concatenate([out_rgb, out_mask], 2) * 255).astype(np.uint8)

image_sam = Image.fromarray(out_img, mode='RGBA')
image_sam.save('./output.png')
