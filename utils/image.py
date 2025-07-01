import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import colorsys

@torch.no_grad()
def crop_with_alpha(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    depth: torch.Tensor = None,
    border: int = 800,
    dfactor: int = 8,
):
    """
    Crops rgb/alpha/depth based on the region where alpha > 0, and keeps a specified border
    outside the bounding box. Then pads it to a size that is a multiple of dfactor.

    Args:
        rgb   : [3, H, W]
        alpha : [1, H, W]
        depth : [1, H, W] or None
        border: border_x for left/right, border_y for top/bottom (border_y = border//2)
        dfactor: The factor that the final (H, W) must be divisible by.

    Returns:
        padded_rgb   : [3, H', W'], where H' and W' are multiples of dfactor.
        padded_depth : [1, H', W'] or None
        xxyy         : (x_min, x_max, y_min, y_max) – in the original image coordinate system,
                       including only the symmetric border (without the alignment padding).
    """

    H = rgb.shape[1]
    W = rgb.shape[2]

    mask = (alpha.squeeze() > 0) # [H, W]

    # Make sure there is at least one alpha > 0
    if not mask.any():
        print("No alpha > 0 found in the input alpha map.")
        return rgb, depth, (0, W, 0, H)

    if border < 0:
        return rgb, depth, (0, W, 0, H)

    # Find the bounding box of the mask
    coords = mask.nonzero(as_tuple=True)  # coords[0] is row indices, coords[1] is column indices (mask is [H, W])
    y_min, y_max = coords[0].min().item(), coords[0].max().item() + 1
    x_min, x_max = coords[1].min().item(), coords[1].max().item() + 1

    # Crop the input tensors
    cropped_rgb = rgb[:, y_min : y_max, x_min : x_max]
    # cropped_alpha = alpha[:, y_min : y_max, x_min : x_max]
    cropped_depth = depth[:, y_min : y_max, x_min : x_max] if depth is not None else None

    border_x = border
    border_y = border // 2

    # Pad the rgb, alpha, and depth tensors
    base_pad = (border_x, border_x, border_y, border_y)
    padded_rgb = F.pad(cropped_rgb, base_pad, mode='constant', value=0)
    padded_depth = F.pad(cropped_depth, base_pad, mode='constant', value=0) if depth is not None else None

    # Calculate the remain padding to make the dimensions divisible by dfactor
    _, H_pad, W_pad = padded_rgb.shape
    extra_w = (-W_pad) % dfactor   # == (dfactor - W_pad % dfactor) % dfactor
    extra_h = (-H_pad) % dfactor

    # Pad the rgb, alpha, and depth tensors to make them divisible by dfactor
    extra_pad = (0, extra_w, 0, extra_h)
    padded_rgb = F.pad(padded_rgb, extra_pad, mode='constant', value=0)
    padded_depth = F.pad(padded_depth, extra_pad, mode='constant', value=0) if depth is not None else None

    # Update the bounding box coordinates in the original image
    x_min = x_min - border_x
    x_max = x_max + border_x + extra_w
    y_min = y_min - border_y
    y_max = y_max + border_y + extra_h

    return padded_rgb, padded_depth, (x_min, x_max, y_min, y_max)


def restore_coords(coords: np.ndarray, xxyy: tuple[int]):
    """
    restore coords to original coordinates
    Input:
        coords: [N, 2] each row is (u, v)
        xxyy: x_min, x_max, y_min, y_max
    Output:
        restored_coords: [N, 2] each row is (u, v)
    """
    x_min, x_max, y_min, y_max = xxyy
    restored_coords = coords + np.array([[x_min, y_min]])
    return restored_coords

def show_feature_matches(
    coords1,
    coords2,
    image1: Image.Image,
    image2: Image.Image,
    save_path,
    x_size=5,
    line_width=2,
):
    # Check if the number of coordinate pairs is consistent
    if len(coords1) != len(coords2):
        raise ValueError("The lengths of coords1 and coords2 must be the same")
    n_viz = len(coords1)

    # Get image dimensions
    width1, height1 = image1.size  # width and height of image1
    width2, height2 = image2.size  # original width and height of image2

    image1.save(save_path.replace(".png", "_image1.png"))
    image2.save(save_path.replace(".png", "_image2.png"))

    # Calculate scaling factors to make image2 match the size of image1
    scale_x = width1 / width2 if width2 != 0 else 1
    scale_y = height1 / height2 if height2 != 0 else 1

    # Resize image2
    image2_resized = image2.resize((width1, height1))

    # Create a new image, horizontally concatenating image1 and image2, with an initial transparent background
    new_img = Image.new('RGBA', (2 * width1, height1), (0, 0, 0, 0))
    new_img.paste(image1.convert('RGBA'), (0, 0))  # Paste image1
    new_img.paste(image2_resized.convert('RGBA'), (width1, 0))  # Paste the resized image2

    # Create a drawing object
    draw = ImageDraw.Draw(new_img)

    # Draw connecting lines and "X" markers
    for i in range(n_viz):
        # Get coordinate pairs
        (x1, y1), (x2, y2) = coords1[i], coords2[i]

        # Scale the coordinates of coords2
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y

        # Round to integer coordinates
        x1_int = int(round(x1))
        y1_int = int(round(y1))
        x2_int = int(round(x2_scaled + width1))  # image2 is on the right, offset by width1
        y2_int = int(round(y2_scaled))

        # Generate color (based on jet colormap)
        hue = i / (n_viz - 1) if n_viz > 1 else 0
        color_float = colorsys.hsv_to_rgb(hue, 1, 1)
        color = tuple(int(c * 255) for c in color_float) + (255,)  # RGBA format

        # Draw connecting line
        draw.line([(x1_int, y1_int), (x2_int, y2_int)], fill=color, width=line_width)

        # Draw "X" marker at (x1_int, y1_int)
        draw.line([(x1_int - x_size, y1_int - x_size), (x1_int + x_size, y1_int + x_size)], fill=color, width=line_width)
        draw.line([(x1_int - x_size, y1_int + x_size), (x1_int + x_size, y1_int - x_size)], fill=color, width=line_width)

        # Draw "X" marker at (x2_int, y2_int)
        draw.line([(x2_int - x_size, y2_int - x_size), (x2_int + x_size, y2_int + x_size)], fill=color, width=line_width)
        draw.line([(x2_int - x_size, y2_int + x_size), (x2_int + x_size, y2_int - x_size)], fill=color, width=line_width)

    # Set pixels where RGB is all 0 to transparent
    # pixels = new_img.load()
    arr = np.array(new_img)
    mask = (arr[..., 0] == 0) & (arr[..., 1] == 0) & (arr[..., 2] == 0)
    arr[mask, 3] = 0
    new_img = Image.fromarray(arr, mode='RGBA')

    # Save the result
    new_img.save(save_path)
