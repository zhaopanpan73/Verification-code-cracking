# coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# conda install Pillow
from PIL import Image
import random
# pip install captcha 安装验证码库
from captcha.image import ImageCaptcha
from VertificationNetwork import IMAGE_HEIGHT ,IMAGE_WIDTH,MAX_CAPTCHA,CHAR_SET_LEN

# captcha.image 介绍 http://www.spiderpy.cn/blog/detail/32
# 本代码生成验证码图片
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

char_set=number + alphabet + ALPHABET
def random_captcha_text(char_set=number + alphabet + ALPHABET, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image():
    # 构造captcha对象
    image = ImageCaptcha()

    captcha_text = random_captcha_text()
    # list->string
    captcha_text = ''.join(captcha_text)
    # 生成图像验证码  image.genrate()产生的是PIL的image对象，此时captcha为PIL对象
    captcha = image.generate(captcha_text)
    image.write(captcha_text, "F:/验证码生成数据库/"+captcha_text + '.jpg')

    captcha_image = Image.open(captcha)
    # 转换为numpu array格式
    captcha_image = np.array(captcha_image)
    # 返回Label和验证码
    return captcha_text, captcha_image

def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return

def text2vec(text):
    # 边界条件检查
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    """
     # 0---9  A---Z  a--z
     # 0---9 10---35 36--61
     ord(c)返回字符c的ASCII
     chr(70)函数是输入一个整数[0，255]返回其对应的ascii符号   
     """
    def char2pos(c):   
        if c =='_':   # 生成的字符以_结尾
            k = 62
            return k   
        k = ord(c)-48   
        if k > 9:   
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:   
                    raise ValueError('No Map')    
        return k   

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


# 向量转回文本
def vec2text(vec):
    """
    # 0---9  A---Z  a--z
    # 0---9 10---35 36--61
    """

    char_pos = vec.nonzero()[0]    # 找出非零的下标
    text=[]   
    for i, c in enumerate(char_pos):   
        char_at_pos = i #c/63   
        char_idx = c % CHAR_SET_LEN   
        if char_idx < 10:   
            char_code = char_idx + ord('0')   
        elif char_idx <36:   
            char_code = char_idx - 10 + ord('A')   
        elif char_idx < 62:   
            char_code = char_idx-  36 + ord('a')   
        elif char_idx == 62:   
            char_code = ord('_')   
        else:   
            raise ValueError('error')   
        text.append(chr(char_code))
    return "".join(text)



#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有  
# vec = text2vec("F5Sd")
# text = vec2text(vec)
# print(text)  # F5Sd
# vec = text2vec("SFd5")
# text = vec2text(vec)
# print(text)  # SFd5


# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)  # 转化为灰度图

        (image.flatten()-128)/128  #  mean为0  简单的方法 batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y




if __name__ == '__main__':
    for i in range(10000):
        text, image = gen_captcha_text_and_image()

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
        plt.imshow(image)

        plt.show()