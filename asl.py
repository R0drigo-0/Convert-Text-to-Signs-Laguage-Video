import os
import re
import json
import math
import requests
import traceback
import nltk
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import concatenate_videoclips
from pytube import YouTube
from nltk.stem import WordNetLemmatizer
from concurrent.futures import ThreadPoolExecutor

# Download necessary NLTK data
nltk.download("wordnet")
nltk.download("omw-1.4")
lemmatizer = WordNetLemmatizer()

# Function Definitions

def read_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file: {e}")
        return None

def parse_srt_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print("SRT file not found.")
        return None

    subtitle_data = []
    word = ""
    start_time_srt = ""
    end_time_srt = ""

    for line in lines:
        line = line.strip()
        if re.match(r"^\d+$", line):
            if word:
                subtitle_data.append((word.strip(), start_time_srt, end_time_srt))
                word = ""
            continue
        elif "-->" in line:
            times = re.findall(r"\d{2}:\d{2}:\d{2},\d{3}", line)
            start_time_srt, end_time_srt = times
            continue
        elif line == "":
            continue
        else:
            word += " " + line

    if word:
        subtitle_data.append((word.strip(), start_time_srt, end_time_srt))

    return subtitle_data

def download_youtube_video(url, output_path):
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(only_video=True, file_extension="mp4").first()
        if stream:
            stream.download(output_path=os.path.dirname(output_path), filename=os.path.basename(output_path))
            return True
        else:
            print(f"No video stream found for URL: {url}")
            return False
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        traceback.print_exc()
        return False

def crop_center_square_and_resize(clip, size=(512, 512)):
    # Crop the video to a 1:1 aspect ratio (square format) and resize to specified size
    width, height = clip.size
    if width != height:
        min_dimension = min(width, height)
        x_center = width / 2
        y_center = height / 2
        x1 = x_center - min_dimension / 2
        y1 = y_center - min_dimension / 2
        x2 = x_center + min_dimension / 2
        y2 = y_center + min_dimension / 2
        clip = clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)
    return clip.resize(size)

def process_video(input_path, output_path, start_time, end_time):
    try:
        print(f"Processing video from {start_time} to {end_time}")
        clip = VideoFileClip(input_path).subclip(start_time, end_time)
        clip = crop_center_square_and_resize(clip).set_fps(30)
        clip.write_videofile(output_path, codec="libx264", fps=30)
        clip.close()
        return True
    except Exception as e:
        print(f"Error processing video: {e}")
        traceback.print_exc()
        return False

def attempt_download_video(word, asl_data, prev_word=None, next_word=None, is_base_form=False):
    word_occurrences = [entry for entry in asl_data if entry.get("clean_text") == word]

    if len(word_occurrences) == 0:
        if not is_base_form:
            base_word = get_synonym(word, prev_word, next_word)
            if base_word != word:
                return attempt_download_video(base_word, asl_data, prev_word, next_word, is_base_form=True)
        return None

    for item in word_occurrences:
        url = item.get("url")
        start_time = item.get("start_time")
        end_time = item.get("end_time")

        if url and start_time is not None and end_time is not None:
            video_name = f"videos/{word}.mp4"
            download_path = os.path.join(os.getcwd(), video_name)

            if download_youtube_video(url, download_path):
                output_path = f"videos/{word}_processed.mp4"
                if process_video(download_path, output_path, start_time, end_time):
                    return output_path

    return None

def get_base_form(word):
    return lemmatizer.lemmatize(word)

def get_synonym(word, prev_word=None, next_word=None):
    url = f"https://api.datamuse.com/words?rel_syn={word}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        synonyms = response.json()
        if not synonyms:
            return word

        synonym_list = [item["word"] for item in synonyms]

        if prev_word or next_word:
            context_url = f"https://api.datamuse.com/words?rel_syn={word}"
            if prev_word:
                context_url += f"&lc={prev_word}"
            if next_word:
                context_url += f"&rc={next_word}"

            context_response = requests.get(context_url)
            context_response.raise_for_status()
            context_synonyms = context_response.json()
            if context_synonyms:
                synonym_list = [item["word"] for item in context_synonyms]

        print(f"SYNONIM: {synonym_list[0]}")
        return synonym_list[0]

    except requests.RequestException as e:
        print(f"Error fetching synonyms: {e}")
        return word

def process_subtitle_data(word, asl_data):
    word = word.replace(" ", "").lower()
    return attempt_download_video(word, asl_data)

if __name__ == "__main__":
    json_file_path = "asl_videos.json"
    asl_data = read_json_file(json_file_path)

    srt_file_path = "captions.srt"
    subtitle_data = parse_srt_file(srt_file_path)

    if subtitle_data and asl_data:
        if not os.path.exists("videos"):
            os.makedirs("videos")

        video_clips = []
        video_paths = []
        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda x: process_subtitle_data(x[0], asl_data), subtitle_data)
            for result in results:
                if result:
                    video_path = result
                    video_clip = VideoFileClip(video_path).set_fps(30)
                    video_clips.append(video_clip)
                    video_paths.append(video_path)

        if video_clips:
            final_clip = concatenate_videoclips(video_clips, method="compose")
            final_clip.write_videofile("concatenated_video.mp4", codec="libx264", fps=30)
            final_clip.close()

        for trimmed_video_path in video_paths:
            os.remove(trimmed_video_path)
    else:
        print("No subtitle data found or ASL data is missing.")
