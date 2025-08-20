import os
import cv2
import Equirec2Perspec as E2P

def process_panorama(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件是否是PNG或JPG格式
        print(filename)
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_folder, filename)
            # 加载等距圆柱投影图像
            equ = E2P.Equirectangular(input_path)

            # 文件名（无扩展名）用于创建子文件夹
            base_filename = os.path.splitext(filename)[0]
            # 生成子文件夹路径，这里假设文件夹已经是以数字序号命名
            output_subfolder = os.path.join(output_folder, base_filename)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            # 定义输出图像的尺寸
            output_height = 512
            output_width = 512

            # 定义六个视角的参数：FOV, theta, phi
            views = {
                '0': (91, 0, 90),
                '1': (91, 0, -90),
                '2': (91, 0, 0),
                '3': (91, 90, 0),
                '4': (91, 180, 0),
                '5': (91, 270, 0)
            }

            # 生成并保存每个视角的图像
            for view_name, (fov, theta, phi) in views.items():
                img = equ.GetPerspective(fov, theta, phi, output_height, output_width)
                output_path = os.path.join(output_subfolder, f'{view_name}.png')
                cv2.imwrite(output_path, img)
            print(f"Processed {filename} and saved views to {output_subfolder}")

if __name__ == '__main__':
    input_folder = '/DATA/DATA3/yl/data/ODI_datasets/matterport'  # 替换为实际的输入文件夹路径
    output_folder = '/DATA/DATA3/yl/data/ODI_datasets/matterport_fov91'  # 替换为实际的输出文件夹路径
    process_panorama(input_folder, output_folder)
    print("done")