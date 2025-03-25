import os
import shutil

# 定义源目录和新的目录结构
base_dir = os.path.expanduser('~/Documents/Develop/AI_CV_Projects/Dir_PoseEstimation/HandPoseShadow')
scripts_dir = os.path.join(base_dir, 'scripts')
assets_dir = os.path.join(base_dir, 'assets')
masks_dir = os.path.join(base_dir, 'masks')

# 创建新目录结构
os.makedirs(scripts_dir, exist_ok=True)
os.makedirs(assets_dir, exist_ok=True)
# masks 目录已经存在，无需创建

# 文件分类规则
script_files = ['demo.py', 'demo_background_v1.py', 'demo_background_v2.py', 
                'detect_hand.py', 'video_demo_mask2.py', 'video_demo_one_hand.py', 
                'video_demo_two_hand.py', 'video_demo_two_hand_distance.py']
image_and_video_files = ['dog1.webp', 'rabbit1.webp', '截屏2024-10-13 00.28.19.png', 
                         '截屏2024-10-13 00.28.27.png']
landmark_files = ['dog_landmarks.npy', 'rabbit_landmarks.npy']

# 移动脚本文件到 scripts 目录
for file in script_files:
    src = os.path.join(base_dir, file)
    if os.path.exists(src):
        shutil.move(src, scripts_dir)
        print(f'Moved {file} to {scripts_dir}')

# 移动图像和视频文件到 assets 目录
for file in image_and_video_files + landmark_files:
    src = os.path.join(base_dir, file)
    if os.path.exists(src):
        shutil.move(src, assets_dir)
        print(f'Moved {file} to {assets_dir}')

# 将 masks 中的图像和视频移动到 assets 目录
masks_src_dir = os.path.join(base_dir, 'masks')
for file in os.listdir(masks_src_dir):
    if file.endswith(('.png', '.mp4', '.jpg', '.webp')):
        src = os.path.join(masks_src_dir, file)
        dest = os.path.join(assets_dir, file)
        shutil.move(src, dest)
        print(f'Moved {file} from masks to {assets_dir}')
