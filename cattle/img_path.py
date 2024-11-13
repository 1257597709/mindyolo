import os

# 定义图片文件夹路径和输出文件路径
# image_folder = 'train/images'
image_folder = 'valid/images'
output_file = 'image_paths.txt'

# 获取图片文件的相对路径
with open(output_file, 'w') as f:
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            # 检查文件是否为图片类型（假设是jpg, png, jpeg格式的图片）
            if file.endswith(('.jpg', '.jpeg', '.png')):
                # 获取文件相对路径
                relative_path = os.path.join(root, file)
                # 将相对路径写入txt文件
                f.write(relative_path + '\n')

print(f"图片路径已保存到 {output_file}")
