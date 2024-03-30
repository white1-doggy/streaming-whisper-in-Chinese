import numpy
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment


def load_audio_binary(file_path):
    with open(file_path, 'rb') as f:
        audio_binary = f.read()
    audio_array = np.frombuffer(audio_binary, dtype=np.int16)
    return audio_array


def calculate_instantaneous_energy(audio_array, window_size=1024):
    squared_frames = audio_array.astype(np.int32) ** 2
    energy = np.convolve(squared_frames, np.ones(window_size) / window_size, mode='valid')
    return energy

# 根据输入语音来源的不同，需要修改 threshold
def detect_pause(energy, threshold=50000):
    pauses = energy < threshold
    return pauses


def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_in_seconds = len(audio) / 1000  # 将毫秒转换为秒
    return duration_in_seconds


def clear_data(data):
    n = 1
    last = 0
    sum = 0
    clear_data = []
    for i in data:
        if i - last <= 0.11 and i - last >= 0.09:
            n = n + 1
            sum = sum + i
            last = i
        else:
            clear_data.append(sum / n)
            sum = i
            n = 1
            last = i
    return clear_data


def pause(bin_data, sample_rates):
    len = bin_data.size / sample_rates
    energy = calculate_instantaneous_energy(bin_data)
    pauses = detect_pause(energy)
    length = energy.size
    # print((np.where(pauses)[0]/length)*len)
    data = (np.where(pauses)[0] / length) * len

    # 连续数据进行清理
    rounded_data = np.around(data, decimals=1)
    unique_elements, counts = np.unique(rounded_data, return_counts=True)

    a = clear_data(unique_elements)

    # a = np.round(a).astype(int)

    # 将时间整理回二进制数据
    pause_net = numpy.multiply(a, length)
    pause_net = numpy.divide(pause_net, len)
    pause_net = np.round(pause_net).astype(int)
    print(pause_net)
    return pause_net


def table(bin_data, sample_rates):
    energy = calculate_instantaneous_energy(bin_data)
    pause_net = pause(bin_data, sample_rates)
    # print(type(pause_net))
    plt.figure(figsize=(10, 4))
    plt.plot(bin_data, label='Audio Signal', color='blue')
    plt.plot(energy, label='Instantaneous Energy', color='red', alpha=0.7)
    plt.scatter(pause_net, bin_data[pause_net], color='black', marker='o', label='Pause')

    plt.title('Audio Signal and Instantaneous Energy')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude / Energy')
    plt.legend()
    plt.show()

# 需要注释掉以下两行
# audio_array = load_audio_binary('split.wav')
# table(audio_array, 16000)
