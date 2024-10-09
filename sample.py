from transformers import AutoProcessor, BarkModel

import torch
import os
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoProcessor, BarkModel
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

# Move the model to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

voice_preset = "v2/ko_speaker_2"


text = """
노부나가는 혁신적인 전술과 무기를 도입해 일본 통일의 기반을 다졌습니다. 특히 그는 유럽에서 전래된 화기를 적극적으로 사용하며, 전통적인 일본 전투 방식에 큰 변화를 가져왔습니다. 그의 전술적 통찰력은 사무라이들이 단순한 무력의 상징이 아닌, 전략적 사고를 갖춘 군사 지도자로서의 면모를 보여주었습니다.
도요토미 히데요시는 오다 노부나가의 뒤를 이어 일본을 통일하는 데 성공했으며, 사무라이 계급의 정치적 힘을 더욱 강화했습니다. 그의 치세 동안 사무라이들은 단순한 전사가 아닌, 국가 운영의 중요한 축으로 자리 잡게 되었습니다.
도쿠가와 이에야스는 에도 시대(江戸時代, Edo period, 1603년 ~ 1868년)를 여는 도쿠가와 막부의 창시자로, 사무라이 계급의 구조를 다시 한 번 재편성했습니다. 그의 치세 동안 사무라이 계급은 사회의 상류층으로 자리 잡았고, 이후 에도 시대에는 상대적으로 평화로운 시대를 맞이하게 되면서 사무라이들의 역할이 다시 변화하기 시작합니다.

"""

inputs = processor(text, voice_preset=voice_preset, return_tensors="pt")

# 수동으로 attention_mask 생성
input_ids = inputs['input_ids']
attention_mask = (input_ids != processor.tokenizer.pad_token_id).long()
inputs['attention_mask'] = attention_mask

# 모든 텐서를 동일한 장치로 이동
inputs = {k: v.to(device) for k, v in inputs.items()}

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

from IPython.display import Audio
# audio_array를 CPU로 이동한 후 다시 device로 이동하는 것은 불필요합니다.
# audio_array = audio_array.to(device)  # 모든 텐서를 동일한 장치로 이동

sample_rate = model.generation_config.sample_rate
Audio(audio_array, rate=sample_rate)

import scipy

# 현재 시간을 가져와서 파일 이름에 추가
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"bark_out_{timestamp}.wav"

sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write(filename, rate=sample_rate, data=audio_array)