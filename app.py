import torch
import torchaudio
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from asteroid.models import BaseModel
import gradio as gr
import os
import uuid

# Load pretrained ConvTasNet model
print("Loading model...")
model = BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
print("Model loaded successfully ✅")

def denoise_and_visualize(audio_path):
    if audio_path is None:
        return "Please upload an audio file.", None, None, None

    try:
        # Unique ID to avoid overwriting files
        uid = str(uuid.uuid4())
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        # Load & resample input to 16kHz mono
        wav, sr = torchaudio.load(audio_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.mean(dim=0, keepdim=True).to(device)

        # Model inference
        with torch.no_grad():
            est_sources = model.separate(wav)
        clean_audio = est_sources[:, 0, :].cpu().squeeze().numpy()

        # Save output audio
        audio_output = os.path.join(output_dir, f"cleaned_{uid}.wav")
        sf.write(audio_output, clean_audio, 16000)

        # Create spectrograms
        orig, _ = librosa.load(audio_path, sr=sr)
        den, _ = librosa.load(audio_output, sr=16000)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(orig)), ref=np.max)
        librosa.display.specshow(D_orig, sr=sr, y_axis='log', x_axis='time')
        plt.title("Original Noisy")
        plt.colorbar(format='%+2.0f dB')

        plt.subplot(1, 2, 2)
        D_clean = librosa.amplitude_to_db(np.abs(librosa.stft(den)), ref=np.max)
        librosa.display.specshow(D_clean, sr=16000, y_axis='log', x_axis='time')
        plt.title("Denoised Output")
        plt.colorbar(format='%+2.0f dB')

        plt.tight_layout()
        spectrogram_output = os.path.join(output_dir, f"spectrogram_{uid}.png")
        plt.savefig(spectrogram_output)
        plt.close()

        return "✅ Denoising complete!", audio_output, spectrogram_output, (16000, clean_audio)

    except Exception as e:
        return f"Error processing audio: {e}", None, None, None

# Gradio UI
iface = gr.Interface(
    fn=denoise_and_visualize,
    inputs=gr.Audio(type="filepath", label="Upload Noisy Audio"),
    outputs=[
        gr.Textbox(label="Status"),
        gr.Audio(label="Denoised Audio"),
        gr.Image(label="Spectrogram Comparison"),
        gr.Audio(label="Denoised Audio (16kHz)"),
    ],
    title="ConvTasNet AI Audio Denoiser",
    description="Upload a noisy audio file. This app removes background noise using ConvTasNet. Spectrograms show before & after.",
)

iface.launch()
