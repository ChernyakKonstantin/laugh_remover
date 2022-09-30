import os

import torch
from lib.models import DCUnet20
from lib.utils import AudioProcessor, save_audio_file
from tqdm import tqdm

model_weights_path = r"./pretrained_weights/Noise2Noise/white.pth"
data_path = r"./audio"
result_path = r"./cleaned_audio"

SAMPLE_RATE = 48000
N_FFT = (SAMPLE_RATE * 64) // 1000 
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000 

def get_deivce():
    train_on_gpu=torch.cuda.is_available()
    if(train_on_gpu):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')
    return torch.device('cuda' if train_on_gpu else 'cpu')



def main() -> None:
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    device = get_deivce()
    dcunet20 = DCUnet20(N_FFT, HOP_LENGTH).to(device)
    checkpoint = torch.load(model_weights_path, map_location=torch.device('cpu'))
    dcunet20.load_state_dict(checkpoint)
    _ = dcunet20.eval()
    audio_processor = AudioProcessor(N_FFT, HOP_LENGTH)
    fnames = list(filter(lambda x: x[-4:] == ".wav", os.listdir(data_path)))
    for fname in tqdm(fnames):
        x, last_len = audio_processor(os.path.join(data_path, fname))
        with torch.no_grad():
            y_hat = dcunet20(x, is_istft=True).cpu()
            save_audio_file(y_hat, last_len, os.path.join(result_path, fname), SAMPLE_RATE)


if __name__ == "__main__":
    main()
