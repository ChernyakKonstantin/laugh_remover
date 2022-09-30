from typing import List, Tuple, Union

import numpy as np
import torch
import torchaudio


def save_audio_file(
    audio: torch.Tensor,
    last_len: int = None,
    file_path: str = './sample_audio.wav',
    sample_rate: int = 48000,
    bit_precision: int = 16
) -> None:
    chunks = list(audio)
    # Remove fillers from the last chunk if they exist
    if last_len is not None:
        chunks[-1] = chunks[-1][-last_len:]
    # Collect track and add channnel dim
    audio = torch.concat(chunks).unsqueeze(0)
    torchaudio.save(file_path, audio, sample_rate,bits_per_sample=bit_precision)


class AudioProcessor:
    """
    Sample processor
    """

    def __init__(self, n_fft: int = 64, hop_length: int = 16, sample_max_len: int = 165000) -> None:
        self.sample_max_len = sample_max_len
        # stft parameters
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, filename: str) -> Tuple[List[torch.Tensor], Union[int, None]]:
        # load to tensors and normalization
        sample = self.load_sample(filename)
        # padding/cutting
        chunks, last_len = self._split_sample(sample)

        def f(x):
            return torch.stft(
                input=x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                normalized=True,
            )
        chunks = list(map(f, chunks))
        chunks = torch.concat(list(map(lambda x: torch.unsqueeze(x, 0), chunks)), dim=0)
        return chunks, last_len

    def load_sample(self, file) -> torch.Tensor:
        waveform, _ = torchaudio.load(file)
        # Convert to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def _split_sample(self, waveform: torch.Tensor) -> Tuple[List[torch.Tensor], Union[int, None]]:
        chunks = list(torch.split(waveform, self.sample_max_len, dim=-1))
        last_len = None
        if chunks[-1].shape[-1] < self.sample_max_len:
            output = torch.zeros((1, self.sample_max_len), dtype=torch.float32)
            last_len = chunks[-1].shape[-1]
            output[:, -last_len:] = chunks[-1]
            chunks[-1] = output
        return chunks, last_len
