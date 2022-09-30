import moviepy.editor
import os


if __name__ == "__main__":
    video_folder = "./video"
    audio_folder = "./audio"

    if not os.path.exists(audio_folder):
        os.mkdir(audio_folder)
    
    filenames = list(filter(lambda x: x[-4:] == ".mp4", os.listdir(video_folder)))
    for filename in filenames:
        video = moviepy.editor.VideoFileClip(os.path.join(video_folder, filename))
        audio = video.audio
        audio.write_audiofile(os.path.join(audio_folder, filename[:-4] + ".wav"))


