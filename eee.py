import wave

def get_wav_info(file_path):
    with wave.open(file_path, 'rb') as wave_file:
        channels = wave_file.getnchannels()
        sample_rate = wave_file.getframerate()
        duration = wave_file.getnframes() / float(sample_rate)

    return channels, sample_rate, duration

file_path = 'stand.wav'
channels, sample_rate, duration = get_wav_info(file_path)

print(f'Channels: {channels}')
print(f'Sample Rate: {sample_rate} Hz')
print(f'Duration: {duration} seconds')
