# Import necessary libraries and configure settings
import random
import re

import ChatTTS
import numpy
import torch
import torchaudio
from IPython.display import Audio

from ai_env.constants import MODEL_DIR

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

BASE = 65248
# Initialize and load the model:
chat = ChatTTS.Chat()
chat.load(source='custom',
          compile=False,
          custom_path=f'{MODEL_DIR}/ChatTTS',
          device='cpu')  # Set to True for better performance


def full_to_half(s):
    # ａＡｚＺ０９
    r = re.compile(r'[ａ-ｚＡ-Ｚ０-９]')
    return r.sub(lambda x: chr(ord(x.group(0)) - BASE), s)


def num2cn(num, traditional=False):
    if traditional:
        chinese_num = {0: '零', 1: '壹', 2: '貳', 3: '叄', 4: '肆', 5: '伍', 6: '陆', 7: '柒', 8: '捌', 9: '玖'}
        chinese_unit = ['仟', '', '拾', '佰']
    else:
        chinese_num = {0: '零', 1: '一', 2: '二', 3: '三', 4: '四', 5: '五', 6: '六', 7: '七', 8: '八', 9: '九'}
        chinese_unit = ['千', '十', '百']
    extra_unit = ['', '万', '亿']
    num_list = list(num)
    num_cn = []
    zero_num = 0  # 连续0的个数
    prev_num = ''  # 遍历列表中当前元素的前一位
    length = len(num_list)

    for num in num_list:
        tmp = num
        if num == '0':  # 如果num为0,记录连续0的数量
            zero_num += 1
            num = ''
        else:
            zero = ''
            if zero_num > 0:
                zero = '零'
            zero_num = 0
            # 处理前一位数字为0，后一位为1，并且在十位数上的数值读法
            if prev_num in ('0', '') and num == '1' and chinese_unit[length % 4] in ('十', '捨'):
                num = zero + chinese_unit[length % 4]
            else:
                num = zero + chinese_num.get(int(num)) + chinese_unit[length % 4]
        if length % 4 == 1:  # 每隔4位加'万'、'亿'拼接
            if num == '零':
                num = extra_unit[length // 4]
            else:
                num += extra_unit[length // 4]
        length -= 1
        num_cn.append(num)
        prev_num = tmp
    num_cn = ''.join(num_cn)
    return num_cn


def arabic_to_chinese(num, traditional=False):
    """
    将阿拉伯数字转换为中文数字

    Args:
        num: 要转换的阿拉伯数字
        traditional: 繁体

    Returns:
        转换后的中文数字字符串
    """
    if traditional:
        chinese_digits = ['零', '壹', '貳', '叄', '肆', '伍', '陆', '柒', '捌', '玖']
        chinese_units = ['', '拾', '佰', '仟', '萬', '拾萬', '百萬', '仟萬', '亿', '拾亿']
    else:
        chinese_digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
        chinese_units = ['', '十', '百', '千', '万', '十万', '百万', '千万', '亿', '十亿']

    if num == 0:
        return '零'

    result = []
    flag = 0  # 用于处理连续的零
    for i, digit in enumerate(reversed(str(num))):
        digit = int(digit)
        if digit != 0:
            result.append(chinese_digits[digit])
            result.append(chinese_units[i])
            flag = 0
        else:
            if flag == 0:
                result.append(chinese_digits[digit])
            flag = 1

    return ''.join(reversed(result))


def generate_seed():
    return {
        "__type__": "update",
        "value": random.randint(1, 100_000_000)
    }


def generate_audio(txt, stream=False):
    rand_spk = chat.sample_random_speaker()
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=rand_spk,  # add sampled speaker
        temperature=.3,  # using custom temperature
    )
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt='[oral_2][laugh_0][break_6]',
    )

    pat_num = '\d+'
    txt = full_to_half(txt)
    matches = re.findall(pattern=pat_num, string=txt)

    for m in matches:
        txt = txt.replace(m, num2cn(m), 1)

    return chat.infer(text=[txt],
                      stream=stream,
                      params_refine_text=params_refine_text,
                      params_infer_code=params_infer_code,
                      use_decoder=True)


def save_audio(name: str, data: numpy.ndarray, by_torch: bool = True):
    if by_torch:
        torchaudio.save(uri=name, src=torch.from_numpy(data), sample_rate=24000)
    else:
        audio = Audio(data, rate=24_000, autoplay=True)
        # f1: save via audio
        with open(file=name, mode="wb+") as f:
            f.write(audio.data)


def output_default(text):
    wavs = generate_audio(text)
    save_audio(name='output.wav', data=wavs[0], by_torch=False)


def output_stream(text):
    wavs = numpy.ndarray([1, 1])

    # Perform inference and play the generated audio
    resp = generate_audio(txt=text, stream=True)

    for w in resp:
        if w[0] is None:
            continue
        wavs = numpy.append(arr=wavs, values=w[0])
    wavs = wavs.reshape([1, wavs.size])
    # f2: save via torch
    save_audio(name='output.wav', data=wavs, by_torch=False)


# Define the text input for inference (Support Batching)
output_stream("据外交部网站消息，2024年7月24日，中共中央政治局委员、外交部长王毅在广州同乌克兰外长库列巴举行会谈。")
