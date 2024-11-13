import os
import json
from PIL import Image
# 定义输入输出路径
labels_dir = 'labels'  # 标签文件所在文件夹
images_dir = 'images'  # 图片文件夹
output_json = 'instances.json'  # 输出的COCO格式JSON文件

# 初始化COCO格式的字典
coco_format = {
    "images": [],
    "annotations": [],
    "categories": []
}
names = [
    'n2005', 's114', 's1547', 's1556', 's1557', 's1607', 's1641', 's1706',
    's1774', 's1778', 's1797', 's1804', 's1805', 's1806', 's1843', 's1853',
    's1854', 's1868', 's1870', 's1877', 's1896', 's1903', 's1906', 's1914',
    's1918', 's1935', 's2011', 's377', 's5562', 's7704', 's8803', 's8804',
    's8813', 's8816', 's8819', 's8832', 's8835', 's8843', 'sn112', 'sn13',
    'sn15', 'sn176', 'sn19', 'sn20', 'sn22', 'sn23'
]

# 生成类别字典
categories = [{"id": idx, "name": name} for idx, name in enumerate(names)]
# 添加类别信息，假设类别从0开始，可以自行根据需求调整  # 类别字典
coco_format["categories"] = categories

annotation_id = 1  # 初始化标注ID

# 遍历labels文件夹中的所有txt文件
for idx, label_file in enumerate(os.listdir(labels_dir)):

    
    
    image_filename = label_file  # 假设图片的扩展名是.jpg
    image_filename = image_filename.replace('.txt', '.jpg')
    image_id = image_filename.replace('.jpg', '')  # 为每张图片分配唯一的ID
    # 假设图片的宽度和高度是已知的，你可以通过读取图片的方式来获取真实的宽高
    # 这里我们假设图片的宽度和高度是640x640
    image = Image.open('images/'+image_filename)
    image_width, image_height = image.size

    # 将图片信息添加到coco_format["images"]
    coco_format["images"].append({
        "id": image_id,
        "file_name": image_filename,
        "width": image_width,
        "height": image_height
    })

    # 读取每个txt文件中的标签数据
    with open(os.path.join(labels_dir, label_file), 'r') as f:
        for line in f.readlines():
            # 假设每一行的格式为: class_id x_center y_center width height (normalized)
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # 将归一化的坐标和尺寸转换为绝对坐标 (x, y, w, h)
            abs_x = (x_center - width / 2) * image_width
            abs_y = (y_center - height / 2) * image_height
            abs_width = width * image_width
            abs_height = height * image_height

            # 添加每个标注到annotations列表中
            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,  # COCO类别ID
                "bbox": [abs_x, abs_y, abs_width, abs_height],  # COCO格式的边界框 [x, y, width, height]
                "area": abs_width * abs_height,  # 区域面积
                "iscrowd": 0  # 假设不是crowd标注
            })
            annotation_id += 1

# 将结果保存为JSON文件
with open(output_json, 'w') as f:
    json.dump(coco_format, f, indent=4)

print(f"COCO格式的JSON文件已保存到 {output_json}")
