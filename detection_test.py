import numpy as np
from PIL import Image
from skimage.io import imsave

from app import white_background
from ldm.util import add_margin
from sam_utils import sam_out_nosave, sam_init
from rembg import remove

raw_im = Image.open('hf_demo/examples/basket.png')
predictor = sam_init()

raw_im.thumbnail([512, 512], Image.Resampling.LANCZOS)
width, height = raw_im.size
image_nobg = remove(raw_im.convert('RGBA'), alpha_matting=True)
arr = np.asarray(image_nobg)[:, :, -1]
x_nonzero = np.nonzero(arr.sum(axis=0))
y_nonzero = np.nonzero(arr.sum(axis=1))
x_min = int(x_nonzero[0].min())
y_min = int(y_nonzero[0].min())
x_max = int(x_nonzero[0].max())
y_max = int(y_nonzero[0].max())
# image_nobg.save('./nobg.png')

image_nobg.thumbnail([512, 512], Image.Resampling.LANCZOS)
image_sam = sam_out_nosave(predictor, image_nobg.convert("RGB"), (x_min, y_min, x_max, y_max))

# imsave('./mask.png', np.asarray(image_sam)[:,:,3]*255)
image_sam = np.asarray(image_sam, np.float32) / 255
out_mask = image_sam[:, :, 3:]
out_rgb = image_sam[:, :, :3] * out_mask + 1 - out_mask
out_img = (np.concatenate([out_rgb, out_mask], 2) * 255).astype(np.uint8)

image_sam = Image.fromarray(out_img, mode='RGBA')
# image_sam.save('./output.png')
