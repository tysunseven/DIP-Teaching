import os
import uuid
import warnings
import argparse
import typing as tp
import enum
import torch
import numpy as np
import imageio
from PIL import Image
from numpy.typing import NDArray
from face_alignment import FaceAlignment, LandmarksType
from draggan import utils
from draggan.draggan import drag_gan
from draggan import draggan as draggan

import matplotlib.pyplot as plt

device = 'cuda'

SIZE_TO_CLICK_SIZE = {
    1024: 8,
    512: 5,
    256: 2
}

CKPT_SIZE = {
    'stylegan2/stylegan2-ffhq-config-f.pkl': 1024,
}

DEFAULT_CKPT = 'stylegan2/stylegan2-ffhq-config-f.pkl'


warnings.filterwarnings('ignore')

class FaceExpression(enum.Enum):
    CLOSE_EYES = 0
    EXPAND_EYES = 1
    CLOSE_LIPS = 2
    SMILE_MOUTH = 3
    SLIM_FACE = 4

    @staticmethod
    def from_description(description: str) -> tp.Self:
        return {
            'close eyes': FaceExpression.CLOSE_EYES,
            'expand eyes': FaceExpression.EXPAND_EYES,
            'close lips': FaceExpression.CLOSE_LIPS,
            'smile mouth': FaceExpression.SMILE_MOUTH,
            'slim face': FaceExpression.SLIM_FACE,
        }[description.lower()]

def close_eyes(in_points: NDArray) -> NDArray:
    return in_points[[37, 38, 43, 44]], in_points[[41, 40, 47, 46]]

def expand_eyes(in_points: NDArray) -> NDArray:
    out_points: NDArray = in_points.copy()
    out_points[[37, 38]] += 0.75 * (out_points[[37, 38]] - out_points[[41, 40]])
    out_points[[43, 44]] += 0.75 * (out_points[[43, 44]] - out_points[[47, 46]])
    return in_points[[37, 38, 43, 44]], out_points[[37, 38, 43, 44]]

def close_lips(in_points: NDArray) -> NDArray:
    out_points: NDArray = in_points.copy()
    diff: NDArray = (out_points[[63, 62, 61]] - out_points[[65, 66, 67]]).mean(axis=1)
    out_points[[65, 66, 67]] = out_points[[63, 62, 61]]
    out_points[55:58, 0] += 1.0 * diff
    return in_points[[65, 66, 67]], out_points[[65, 66, 67]]

def smile_mouth(in_points: NDArray) -> NDArray:
    out_points: NDArray = in_points.copy()
    diff: float = (out_points[54, 1] - out_points[48, 1]).item()
    out_points[54] += np.array([-diff, diff]) * 0.1
    out_points[48] += diff * -0.1
    out_points[64] += np.array([-diff, diff]) * 0.05
    out_points[60] += diff * -0.05
    return in_points[[54, 48, 64, 60]], out_points[[54, 48, 64, 60]]

def slim_face(in_points: NDArray) -> NDArray:
    out_points: NDArray = in_points.copy()
    diff: NDArray = out_points[0:8] - out_points[9:17]
    out_points[0:8] -= 0.05 * diff
    out_points[9:17] += 0.05 * diff
    return in_points[0:17], out_points[0:17]

def raise_eyebrows(in_points: NDArray) -> NDArray:
    out_points = in_points.copy()
    out_points[17:22] += np.array([0, 0.1]) * 5
    out_points[22:27] += np.array([0, 0.1]) * 5
    return in_points[17:27], out_points[17:27]

def get_transformation(expr_id: FaceExpression) -> tp.Callable[[NDArray], NDArray]:
    return {
        FaceExpression.CLOSE_EYES: close_eyes,
        FaceExpression.EXPAND_EYES: expand_eyes,
        FaceExpression.CLOSE_LIPS: close_lips,
        FaceExpression.SMILE_MOUTH: smile_mouth,
        FaceExpression.SLIM_FACE: slim_face,
    }[expr_id]

def get_landmarks(image: NDArray | Image.Image) -> NDArray:
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    fa = FaceAlignment(landmarks_type=LandmarksType.TWO_D)
    landmarks: list[NDArray] = fa.get_landmarks_from_image(image)
    return landmarks[0][:, ::-1]

def get_expression_points(landmarks: NDArray, expr_id: FaceExpression) -> list[list[float]]:
    start_pts, end_pts = get_transformation(expr_id)(landmarks)
    return start_pts.tolist(), end_pts.tolist()

def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    arr = tensor.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = arr * 255
    return arr.astype('uint8')

def add_points_to_image(image, points, size=3):
    image = utils.draw_handle_target_points(image, points['handle'], points['target'], size)
    return image

def on_drag(model, points, max_iters, state, size, mask, lr_box):
    if len(points['handle']) == 0:
        raise ValueError('You must select at least one handle point and target point.')
    if len(points['handle']) != len(points['target']): 
        raise ValueError('You have uncompleted handle points, try to select a target point or undo the handle point.')
    max_iters = int(max_iters)
    W = state['W']

    handle_points = [torch.tensor(p, device=device).float() for p in points['handle']]
    target_points = [torch.tensor(p, device=device).float() for p in points['target']]

    if mask.get('mask') is not None:
        mask = Image.fromarray(mask['mask']).convert('L')
        mask = np.array(mask) == 255
        mask = torch.from_numpy(mask).float().to(device)
        mask = mask.unsqueeze(0).unsqueeze(0)
    else:
        mask = None

    step = 0
    for image, W, handle_points in drag_gan(W, model['G'], handle_points, target_points, mask, max_iters=max_iters, lr=lr_box):
        points['handle'] = [p.cpu().numpy().astype('int') for p in handle_points]
        image = add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])
        state['history'].append(image)
        step += 1

    os.makedirs('draggan_tmp', exist_ok=True)
    image_name = f'draggan_tmp/image_{uuid.uuid4()}.png'
    video_name = f'draggan_tmp/video_{uuid.uuid4()}.mp4'
    imageio.imsave(image_name, image)
    imageio.mimsave(video_name, state['history'])
    return image, state, step

def on_change_expression(expr_str: str, points: dict[str, tp.Any]) -> dict[str, tp.Any]:
    landmarks: list[list[float]] = points['orig']
    expr_id: FaceExpression = FaceExpression.from_description(expr_str)
    points['handle'], points['target'] = get_expression_points(np.array(landmarks), expr_id)
    return points

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--expression', choices=[
        # 'Close eyes', 'Expand eyes', 'Close lips', 'Smile mouth', 'Slim face'], default='Close eyes')
        'Close eyes', 'Expand eyes', 'Close lips', 'Smile mouth', 'Slim face'], default='Close lips')
    parser.add_argument('--max_iters', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-3)
    args = parser.parse_args(args=[])

    device = args.device
    expression = args.expression
    max_iters = args.max_iters
    lr = args.lr

    torch.cuda.manual_seed(25)

    G = draggan.load_model(utils.get_path(DEFAULT_CKPT), device=device)
    W = draggan.generate_W(
        G,
        seed=int(1),
        device=device,
        truncation_psi=0.8,
        truncation_cutoff=8,
    )
    img, F0 = draggan.generate_image(W, G, device=device)

    state = {
        'W': W,
        'img': img,
        'history': []
    }

    points_dict: dict[str, list[list[float]]] = {}
    landmarks: NDArray = get_landmarks(img)
    points_dict['orig'] = landmarks.tolist()
    points_dict['handle'], points_dict['target'] = get_expression_points(landmarks, FaceExpression.from_description(expression))
    points = points_dict

    size = CKPT_SIZE[DEFAULT_CKPT]

    original_image = add_points_to_image(img, points, size=SIZE_TO_CLICK_SIZE[size])

    mask = {'mask': None}

    image, state, progress = on_drag({'G': G}, points, max_iters, state, size, mask, lr)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image with Points')
    axes[1].imshow(image)
    axes[1].set_title(f'Expression: {expression}')
    plt.show()

if __name__ == '__main__':
    main()