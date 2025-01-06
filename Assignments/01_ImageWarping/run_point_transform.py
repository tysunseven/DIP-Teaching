import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    # 基于MLS的图像变形（Affine Model）
    # source_pts: 控制点原始位置   shape: (N, 2)
    # target_pts: 控制点目标位置   shape: (N, 2)
    # alpha: 控制距离权重衰减程度
    # eps: 防止除零错误的一个小值
    
    if len(source_pts) == 0 or len(source_pts) != len(target_pts):
        # 若无控制点或数量不匹配，则返回原图
        return image.copy()
    
    h, w = image.shape[:2]
    warped_image = np.zeros_like(image, dtype=image.dtype)
    
    source_pts = source_pts.astype(np.float64)
    target_pts = target_pts.astype(np.float64)
    
    for y in range(h):
        for x in range(w):
            X = np.array([x, y], dtype=np.float64)
            
            # 计算权重 w_i = 1 / ||X - p_i||^(2alpha)
            # 若X与某个p_i非常接近，那么直接将X映射到对应的q_i，以减少数值不稳定性
            distances = np.linalg.norm(source_pts - X, axis=1)
            close_idx = np.where(distances < eps)[0]
            if len(close_idx) > 0:
                # 如果点X太接近某个控制点p_i，那就直接映射到对应的q_i
                idx = close_idx[0]
                Xp = target_pts[idx]
                Xp_int = Xp.astype(np.int32)
                if 0 <= Xp_int[0] < w and 0 <= Xp_int[1] < h:
                    warped_image[y, x] = image[Xp_int[1], Xp_int[0]]
                else:
                    warped_image[y, x] = 0
                continue
            
            weights = 1.0 / np.power(distances, 2 * alpha)
            W = np.sum(weights)

            # 计算加权重心
            p_star = np.sum(source_pts * weights[:, None], axis=0) / W
            q_star = np.sum(target_pts * weights[:, None], axis=0) / W

            # 去中心化
            p_prime = source_pts - p_star
            q_prime = target_pts - q_star

            # 计算A矩阵：A = (Σ w_i q_i' p_i'^T) (Σ w_i p_i' p_i'^T)^(-1)
            # p_i'和q_i'为2D向量，因此p_prime, q_prime为(N,2)
            p_weighted = p_prime * weights[:, None]  # (N,2)
            # Σ w_i p_i' p_i'^T
            M = p_weighted.T @ p_prime  # 2x2 矩阵
            # Σ w_i q_i' p_i'^T
            N = (q_prime * weights[:, None]).T @ p_prime  # 2x2 矩阵

            # 当M不可逆时，退化为纯平移
            if abs(np.linalg.det(M)) < eps:
                A = np.eye(2)
            else:
                A = N @ np.linalg.inv(M)

            # 对当前点X进行变换 X' = q_star + A (X - p_star)
            X_prime = q_star + A @ (X - p_star)
            Xp_int = X_prime.astype(np.int32)

            # 最近邻取样
            if 0 <= Xp_int[0] < w and 0 <= Xp_int[1] < h:
                warped_image[y, x] = image[Xp_int[1], Xp_int[0]]
            else:
                # 超出范围的点填充为0或背景值
                warped_image[y, x] = 0

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch(share=True)
