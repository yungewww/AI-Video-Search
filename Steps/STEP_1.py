'''
STEP 1
- Download videos and captions From YouTube
- Instruction
'''

# CHANGE YOUR URLS HERE
urls = [
    "https://www.youtube.com/watch?v=wbWRWeVe1XE",
    "https://www.youtube.com/watch?v=FlJoBhLnqko",
    "https://www.youtube.com/watch?v=Y-bVwPRy_no"
]

from pytube import YouTube
import yt_dlp

def download_video_and_captions(url, path='./'):
    # Download Video
    yt = YouTube(url)
    video = yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first()
    video_path = video.download(output_path=path)

    # Download Captions in 'vtt' format
    ydl_opts = {
        'writesubtitles': True,
        'subtitleslangs': ['en'],  # Download English subtitles
        'writeautomaticsub': True,  # Download automatic captions if available
        'skip_download': True,  # Skip downloading the video again
        'outtmpl': path + '/%(title)s.%(ext)s',  # Output template for the caption file
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    caption_path = path + '/' + yt.title + '.en.vtt'

    return video_path, caption_path

download_path = "./videos"

# DOWNLOAD START
for url in urls:
    video_path, caption_path = download_video_and_captions(url, download_path)
    print(f"Downloaded video from {url}")
    print(f"Video path: {video_path}")
    print(f"Caption path: {caption_path}")