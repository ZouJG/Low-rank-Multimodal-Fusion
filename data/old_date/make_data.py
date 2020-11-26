import pickle
import os
import jieba
import numpy as np
from gensim.models import KeyedVectors

from PIL import Image
import scipy.io.wavfile
import python_speech_features
import codecs

VEC_PATH = "/home/jack/PycharmProjects/LowrankMultimodalFusion/models/Tencent_AILab_ChineseEmbedding.txt"

# jieba.enable_paddle()  # 启动paddle模式。

# word_vec = KeyedVectors.load_word2vec_format(VEC_PATH, binary=False)  # 加载预训练词向量
word_vec = pickle.load(open("/home/jack/PycharmProjects/LowrankMultimodalFusion/models/word_vec.pkl","rb"))
stop_word = [x.replace("\n", "").replace("\r", "") for x in
             codecs.open("/home/jack/PycharmProjects/LowrankMultimodalFusion/data/stop_word.txt", 'r',
                         encoding='utf-8')]

LABEL_NUM = 3

AUDIO = 'covarep'
VISUAL = 'facet'
TEXT = 'glove'
LABEL = 'label'
TRAIN = 'train'
VALID = 'valid'
TEST = 'test'

my_data = {}
test_data = {}
valid_data = {}
train_data = {}


def read_label(label_path):
    file_names = []
    labels = []
    with codecs.open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            file_names.append(line.split(" ")[0])
            labels.append(int(line.split(" ")[1].replace("\n", "")))

    return file_names, labels


def get_audio_file(audio_path, file_names):
    return [audio_path + file_name + "_audio.wav" for file_name in file_names]


def get_text_file(text_path, file_names):
    return [text_path + file_name + "_text.txt" for file_name in file_names]


def get_viode_file(video_path, file_names):
    return [video_path + file_name + "/0.jpg" for file_name in file_names]


def get_files(data_path):
    file_names, labels = read_label(data_path + "/label.txt")

    audio_path = data_path + "/audio/"

    text_path = data_path + "/text/"

    video_path = data_path + "/video/"

    audios = get_audio_file(audio_path, file_names)

    texts = get_text_file(text_path, file_names)

    videos = get_viode_file(video_path, file_names)

    return audios, texts, videos, labels


def text_to_vec(text_path):

    texts = []

    with codecs.open(text_path, 'r', encoding='GB2312') as f:

        for line in f:
            temp = line.split(":")[1].replace("\n", "").replace("\r", "")
            if len(temp) > 2:
                texts.append(temp)

    text_vec = []

    for text in texts:

        seg_list = jieba.cut(text)

        sen_vec = np.zeros(200)
        word_num = 0

        for seg in seg_list:

            if seg not in stop_word:
                try:
                    sen_vec += word_vec[seg]
                    word_num += 1
                except:
                    pass

        if word_num>10:
            sen_vec = sen_vec / word_num
            text_vec.append(sen_vec)

    if len(text_vec)>256:
        text_vec = text_vec[0:256]

    while len(text_vec) <256:
        text_vec.append(np.ones(200))

    return np.array(text_vec)


def audiuo_to_vec(audio_path):
    sample_rate, signal = scipy.io.wavfile.read(audio_path, mmap=False)  # 加载语音文件

    mfcc = python_speech_features.base.mfcc(signal=signal, samplerate=sample_rate, winlen=0.025, winstep=0.01,
                                            numcep=13,
                                            nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22,
                                            appendEnergy=True,
                                            winfunc=lambda x: np.ones((x,)))

    mfcc = np.mean(mfcc, axis=0).reshape(1, -1)[0]

    return mfcc


def read_image(image_path):
    img = Image.open(image_path)
    image = np.array(img)
    image = np.resize(image, 1024)
    return image


def gen_label(label):
    label_array = np.zeros(LABEL_NUM)
    label_array[label - 1] = 1

    return label_array


#
# read_image("/Users/jackzo/PycharmProjects/LowrankMultimodalFusion/predict/pic.jpg")
#
# audiuo_to_vec("/Users/jackzo/PycharmProjects/LowrankMultimodalFusion/data/test.wav")

# text_to_vec("/Users/jackzo/PycharmProjects/LowrankMultimodalFusion/data/text/101_text.txt")

audios, texts, videos, labels = get_files("/home/jack/PycharmProjects/LowrankMultimodalFusion/data")

audio_data = []
text_data = []
video_data = []

for text_path in texts:
    text_data.append(text_to_vec(text_path))

for viode_path in videos:
    video_data.append(read_image(viode_path))

for audio_path in audios:
    audio_data.append(audiuo_to_vec(audio_path))



labels = [gen_label(x) for x in labels]


train_data[VISUAL] = np.array(video_data)
train_data[AUDIO] = np.array(audio_data)
train_data[TEXT] = np.array(text_data)
train_data[LABEL] = np.array(labels)

test_data[TEXT] = np.array(text_data)
test_data[VISUAL] = np.array(video_data)
test_data[AUDIO] = np.array(audio_data)
test_data[LABEL] = np.array(labels)

valid_data[TEXT] = np.array(text_data)
valid_data[VISUAL] = np.array(video_data)
valid_data[AUDIO] = np.array(audio_data)
valid_data[LABEL] = np.array(labels)

my_data["test"] = test_data
my_data["train"] = train_data
my_data["valid"] = valid_data

pickle.dump(my_data, open("my_data.pkl", "wb"))