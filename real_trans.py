import argparse
import os
import numpy as np
from scipy.io.wavfile import write
import speech_recognition as sr
import opencc
from ppasr.infer_utils.pun_predictor import PunctuationPredictor

from datetime import datetime, timedelta
from queue import Queue
from time import sleep

from faster_whisper import WhisperModel
from get_pause import *


def getpause(data_bin, sample_rate):
    flag = True
    pause_time = pause(data_bin, sample_rate)
    if pause_time.size == 1 or pause_time.size == 0:
        flag = False
    else:
        pause_time = pause_time[1:]
    return flag, pause_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic_energy_threshold", default=False, help="Don't use the dynamic_energy_threshold.")
    parser.add_argument("--sample_rate", default=16000, help="Sample rate", type=int)
    parser.add_argument("--record_timeout", default=1, help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--audio_path", default='audio/test2.wav', help="Preheating audio path", type=str)
    parser.add_argument("--model_path", default='whisper-ct2-finetune', help="Whisper model path", type=str)
    parser.add_argument("--pun_model_path", default='pun_models', help="Punctuation model path", type=str)
    parser.add_argument("--language", default='zh', help="Model language", type=str)

    args = parser.parse_args()


    data_queue = Queue()
    data_trans = Queue()
    recorder = sr.Recognizer()
    recorder.dynamic_energy_threshold = args.dynamic_energy_threshold
    source = sr.Microphone(sample_rate=args.sample_rate)
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)
        data_trans.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=args.record_timeout)

    audio_model = WhisperModel(args.model_path, device='cuda', compute_type='auto', cpu_threads=0)
    _ = audio_model.transcribe(args.audio_path, language=args.language)
    pun_predictor = PunctuationPredictor(args.pun_model_path)

    if _ is not None:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Model loaded.\n")

    transcription = ['']
    seg = np.array([])
    last_seg = np.array([])

    while True:
        try:

            if not data_queue.empty():

                audio_data = b''.join(data_queue.queue)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                flag, pause_net = getpause(audio_np, args.sample_rate)

                if flag:
                    len_audio_bin = audio_np.size
                    seg = audio_np[pause_net[-1]:len_audio_bin]
                    audio_np = audio_np[pause_net[-1]:]
                    if last_seg.any():
                        audio_np = np.hstack((last_seg, audio_np))
                    audio_np = audio_np.astype(np.float32) / 32768.0
                    last_seg = seg
                    data_queue.queue.clear()
                else:
                    audio_np = audio_np.astype(np.float32) / 32768.0
                    if last_seg.any():
                        audio_np = np.hstack((last_seg, audio_np))
                    last_seg = np.array([])

                text = ""
                segments, info = audio_model.transcribe(audio_np, language=args.language)
                for segment in segments:
                    text += segment.text

                cc = opencc.OpenCC('t2s')
                text = cc.convert(text)
                text = pun_predictor(text)

                if flag:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                os.system('cls' if os.name == 'nt' else 'clear')

                str_text = "".join(transcription)

                print(str_text)
                print('', end='', flush=True)

                sleep(0.25)

        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    str_text2 = "".join(transcription)
    str_text2 = pun_predictor(str_text2)
    print(str_text2)

    # 将转录的语音保存下来，用于检查自己录音设施是否完整，可以注释
    # audio_data = b''.join(data_trans.queue)
    # audio_np = np.frombuffer(audio_data, dtype=np.int16)
    # output_file = "saved_audio.wav"
    # write(output_file, args.sample_rate, audio_np)
    # print(f"音频已保存到 {output_file}")


if __name__ == "__main__":
    main()
