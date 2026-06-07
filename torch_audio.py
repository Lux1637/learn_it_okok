import os
import torch
import torchaudio
import soundfile as sf
import numpy as np

# ------------------------------------------------------------
# 1. 下载或使用已有的音频文件
# ------------------------------------------------------------
SAMPLE_WAV_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.wav"
SPEECH_FILE = "steam_train.wav"

def download_file(url, dest):
    try:
        import requests
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        with open(dest, 'wb') as f:
            f.write(resp.content)
        print(f"✓ 文件下载成功: {dest} ({len(resp.content)} 字节)")
        return True
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return False

if not os.path.exists(SPEECH_FILE) or os.path.getsize(SPEECH_FILE) == 0:
    print("本地音频文件不存在，正在从网络下载...")
    if not download_file(SAMPLE_WAV_URL, SPEECH_FILE):
        raise RuntimeError("无法下载音频文件")

# ------------------------------------------------------------
# 2. 使用 soundfile 读取音频（不依赖 torchaudio 后端）
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

try:
    # soundfile 读取：返回 (data, sample_rate)
    # data 形状为 (samples, channels)  -> 需要转为 (channels, samples)
    audio_data, sample_rate = sf.read(SPEECH_FILE, dtype='float32')
    if len(audio_data.shape) == 1:
        # 单声道：形状 (samples,) -> (1, samples)
        waveform = torch.from_numpy(audio_data).unsqueeze(0)
    else:
        # 多声道：形状 (samples, channels) -> (channels, samples)
        waveform = torch.from_numpy(audio_data.T)
    print(f"✓ 音频加载成功 (使用 soundfile)")
    print(f"  原始采样率: {sample_rate} Hz")
    print(f"  形状: {waveform.shape} (通道数 x 采样点数)")
    print(f"  时长: {waveform.shape[-1] / sample_rate:.2f} 秒")
except Exception as e:
    print(f"✗ soundfile 读取失败: {e}")
    # 回退方案：使用 scipy.io.wavfile
    try:
        from scipy.io import wavfile
        sample_rate, audio_data = wavfile.read(SPEECH_FILE)
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / 32767.0  # 16-bit PCM 归一化
        if len(audio_data.shape) == 1:
            waveform = torch.from_numpy(audio_data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_data.T)
        print(f"✓ 音频加载成功 (使用 scipy)")
    except Exception as e2:
        print(f"✗ scipy 也失败: {e2}")
        raise RuntimeError("无法加载音频文件，请安装 soundfile: pip install soundfile")

# ------------------------------------------------------------
# 3. 重采样到 16000 Hz（Wav2Vec2 要求）
# ------------------------------------------------------------
TARGET_SR = 16000
if sample_rate != TARGET_SR:
    print(f"重采样: {sample_rate} Hz → {TARGET_SR} Hz")
    resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SR)
    waveform = resampler(waveform)
    sample_rate = TARGET_SR

waveform = waveform.to(device)

# ------------------------------------------------------------
# 4. 加载预训练 Wav2Vec2 模型并推理
# ------------------------------------------------------------
print("加载 Wav2Vec2 模型...")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
model.eval()

with torch.inference_mode():
    emissions, _ = model(waveform)

# ------------------------------------------------------------
# 5. CTC 解码器
# ------------------------------------------------------------
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank
    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emissions[0])
print("\n🎤 识别结果：")
print(transcript)