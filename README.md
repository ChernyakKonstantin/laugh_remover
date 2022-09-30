All scripts from `./scripts` should be run from root directory as follows:

`python -m scripts.SCRIPTNAME`

Scripts:
* `get_audio.py` - obtained audio from video. Path to folders are specified within the script.

* `clean_dataset.py` - clean audio from backgorund noise (laugh, music and etc). Path to folders or weight are specified within the script. 

The audio track is arbitrary lenght. This is the reason the model process one track per run (tracks if splitted into batch to be processed). Probably too large track will no fit into RAM/GPU. Tested with 10 seconds length track on CPU with 16GB RAM. 

Reference: https://github.com/madhavmk/Noise2Noise-audio_denoising_without_clean_training_data