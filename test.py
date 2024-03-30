import argparse
import functools
import os

from faster_whisper import WhisperModel


from datetime import datetime, timedelta

# from ppasr.infer_utils.pun_predictor import PunctuationPredictor

# pun_predictor = PunctuationPredictor('pun_models')



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


model = WhisperModel("whisper-ct2-finetune", device="cuda", compute_type="int8", num_workers=1, local_files_only=True)

# 语音识别
last = datetime.utcnow()
print(last)
segments, info = model.transcribe("stand.wav", beam_size=10, language="zh",
                                  vad_filter=False)
for segment in segments:
    text = segment.text
    # print(f"[{round(segment.start, 2)} - {round(segment.end, 2)}]：{text}\n")
    print(f"{text}")

# for segment in segments:
#     text = segment.text
#     text = pun_predictor(text)
#     # print(f"[{round(segment.start, 2)} - {round(segment.end, 2)}]：{text}\n")
#     print(f"{text}")

# text = pun_predictor(text)
