# Import necessary libraries and configure settings
import torch
import torchaudio

from ai_env.constants import MODEL_DIR

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

import ChatTTS

from IPython.display import Audio

# Initialize and load the model:
chat = ChatTTS.Chat()
chat.load(source='custom',
          compile=False,
          custom_path=f'{MODEL_DIR}/ChatTTS',
          device='cpu')  # Set to True for better performance

# Define the text input for inference (Support Batching)
texts = [
    "据外交部网站消息，2024年7月24日，中共中央政治局委员、外交部长王毅在广州同乌克兰外长库列巴举行会谈。",
]

# Perform inference and play the generated audio
wavs = chat.infer(texts)
audio = Audio(wavs[0], rate=24_000, autoplay=True)

with open(file="output.wav", mode="wb+") as f:
    f.write(audio.data)
# Save the generated audio
# torchaudio.save(uri="output.wav", src=torch.from_numpy(wavs[0]), sample_rate=24000, format='wav')
