from pydub import AudioSegment
import os

# file = wave.open(wave_path)
# # print('---------声音信息------------')
# # for item in enumerate(WAVE.getparams()):
# #     print(item)
# a = file.getparams().nframes  # 帧总数
# f = file.getparams().framerate  # 采样频率
# sample_time = 1 / f  # 采样点的时间间隔
# time = a / f  # 声音信号的长度
# sample_frequency, audio_sequence = wavfile.read(wave_path)
# # print(audio_sequence)  # 声音信号每一帧的“大小”
# x_seq = np.arange(0, time, sample_time)
#
# plt.plot(x_seq, audio_sequence, 'blue')
# plt.xlabel("time (s)")

file_name = ".\\getradio_demo\\陈桂青.wav"
sound = AudioSegment.from_mp3(file_name)
start_time = "0:00"
stop_time = "1:00"
print("time:", start_time, "~", stop_time)
start_time = (int(start_time.split(':')[0]) * 60 + int(start_time.split(':')[1])) * 1000
stop_time = (int(stop_time.split(':')[0]) * 60 + int(stop_time.split(':')[1])) * 1000
print("ms:", start_time, "~", stop_time)
word = sound[start_time:stop_time]
save_name = '.\\' + file_name.split('\\')[-1]
# os.mknod(save_name)
print(save_name)
word.export(save_name, format="wav", tags={'artist': 'AppLeU0', 'album': save_name[:-4]})
