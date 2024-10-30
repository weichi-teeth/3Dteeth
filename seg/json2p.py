import json
from PIL import Image, ImageDraw

# 读取JSON文件
with open('/media/cx206/work/teeth/seg/valid/500RealCases1/51/╓╨╞┌╜╫╢╬1/low.json', 'r') as f:
    data = json.load(f)

# 创建一个黑色背景的图像
image_width = data['imageWidth']
image_height = data['imageHeight']
image = Image.new('RGB', (image_width, image_height), color=(0, 0, 0))  # 黑色背景
draw = ImageDraw.Draw(image)

for i in range(len(data['shapes'])):
    shape = data['shapes'][i]  # 假设我们只关心第一个形状
    points = [tuple(map(int, p)) for p in shape['points']]
    line_color = (255, 255, 255)  # 确保颜色为白色或黑色
    line_width = 3
    draw.line(points, fill=line_color, width=line_width)
    last_point = points[-1]
    first_point = points[0]
    draw.line([last_point, first_point], fill=line_color, width=line_width)

# 保存图像
image.save('/media/cx206/work/teeth/seg/valid/500RealCases1/51/╓╨╞┌╜╫╢╬1/low.png')