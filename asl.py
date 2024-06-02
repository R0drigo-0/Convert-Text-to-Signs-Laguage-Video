import os
import re
import json
import shutil
import requests
import traceback
import spacy
from concurrent.futures import ThreadPoolExecutor
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import concatenate_videoclips
from pytube import YouTube
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

nlp = spacy.load('en_core_web_sm')

def read_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error reading JSON file: {e}")
        return None

def parse_srt_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except FileNotFoundError:
        logging.error("SRT file not found.")
        return None

    words = []

    for line in lines:
        line = line.strip()
        if re.match(r"^\d+$", line) or "-->" in line or line == "":
            continue
        words.extend(line.split())

    return words

def download_youtube_video(url, output_path):
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(only_video=True, file_extension="mp4").first()
        if stream:
            stream.download(
                output_path=os.path.dirname(output_path),
                filename=os.path.basename(output_path),
            )
            return True
        else:
            logging.error(f"No video stream found for URL: {url}")
            return False
    except Exception as e:
        logging.error(f"Error downloading YouTube video: {e}")
        traceback.print_exc()
        return False

def crop_center_square_and_resize(clip, size=(1280, 720)):
    return clip.resize(size)

def process_video(input_path, start_time, end_time):
    try:
        logging.info(f"Processing video from {start_time} to {end_time}")
        clip = VideoFileClip(input_path).subclip(start_time, end_time)
        clip = crop_center_square_and_resize(clip).set_fps(30)
        
        temp_path = input_path + ".temp.mp4"
        clip.write_videofile(temp_path, codec="libx264", fps=30)
        clip.close()
        
        os.replace(temp_path, input_path)
        return True
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        traceback.print_exc()
        return False

def attempt_download_video(word, asl_data, order, prev_word=None, next_word=None, is_base_form=False, is_synonym=False):
    def check_word_in_asl_data(word_to_check):
        return [entry for entry in asl_data if entry.get("clean_text") == word_to_check]

    word_occurrences = check_word_in_asl_data(word)

    if not word_occurrences:
        if not is_base_form:
            base_word = get_base_form(word)
            base_word_occurrences = check_word_in_asl_data(base_word)
            if base_word_occurrences:
                return attempt_download_video(base_word, asl_data, order, prev_word, next_word, is_base_form=True)
        if not is_synonym:
            synonym = get_synonym(word, prev_word, next_word)
            synonym_occurrences = check_word_in_asl_data(synonym)
            if synonym_occurrences:
                return attempt_download_video(synonym, asl_data, order, prev_word, next_word, is_synonym=True)
        if is_synonym and not is_base_form:
            base_word_of_synonym = get_base_form(word)
            base_word_of_synonym_occurrences = check_word_in_asl_data(base_word_of_synonym)
            if base_word_of_synonym_occurrences:
                return attempt_download_video(base_word_of_synonym, asl_data, order, prev_word, next_word, is_base_form=True, is_synonym=True)
        return None

    for item in word_occurrences:
        url = item.get("url")
        start_time = item.get("start_time")
        end_time = item.get("end_time")

        if url and start_time is not None and end_time is not None:
            video_name = f"videos/{order:05d}_{word}.mp4"
            download_path = os.path.join(os.getcwd(), video_name)

            if download_youtube_video(url, download_path):
                if process_video(download_path, start_time, end_time):
                    return download_path

    return None

def get_base_form(word):
    return str(nlp(word)[0].lemma_)

def get_synonym(word, prev_word=None, next_word=None):
    def fetch_synonyms(word):
        url = f"https://api.datamuse.com/words?rel_syn={word}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Error fetching synonyms for {word}: {e}")
            return []

    def fetch_contextual_synonyms(word, prev_word, next_word):
        url = f"https://api.datamuse.com/words?rel_syn={word}"
        if prev_word:
            url += f"&lc={prev_word}"
        if next_word:
            url += f"&rc={next_word}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Error fetching contextual synonyms for {word}: {e}")
            return []

    synonyms = fetch_synonyms(word)
    if not synonyms:
        return word

    synonym_list = [item["word"] for item in synonyms]

    if prev_word or next_word:
        contextual_synonyms = fetch_contextual_synonyms(word, prev_word, next_word)
        if contextual_synonyms:
            return contextual_synonyms[0]["word"]

    return synonym_list[0] if synonym_list else word

def process_subtitle_data(word, asl_data, order):
    word = word.replace(" ", "").lower()
    return attempt_download_video(word, asl_data, order)

def batch_concatenate_clips(video_paths, batch_size, output_path):
    batch_number = 0
    batch_clips = []
    batch_video_paths = []
    for i in range(0, len(video_paths), batch_size):
        batch = video_paths[i:i+batch_size]
        clips = [VideoFileClip(video).set_fps(30) for video in batch]
        batch_clip = concatenate_videoclips(clips, method="compose")
        batch_output_path = f"{output_path}_batch{batch_number:03d}.mp4"
        batch_clip.write_videofile(batch_output_path, codec="libx264", fps=30)
        batch_clips.append(batch_clip)
        batch_video_paths.append(batch_output_path)
        batch_number += 1
    final_clip = concatenate_videoclips(batch_clips, method="compose")
    final_clip.write_videofile(output_path, codec="libx264", fps=30)
    final_clip.close()
    for path in batch_video_paths:
        os.remove(path)

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
            future_to_word = {
                executor.submit(process_subtitle_data, word, asl_data, idx): word
                for idx, word in enumerate(subtitle_data)
            }
            for future in future_to_word:
                result = future.result()
                if result:
                    video_path = result
                    logging.info(f"Processed video path: {video_path}")
                    video_paths.append(video_path)

        video_paths.sort(key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))

        output_path = "concatenated_video.mp4"
        batch_size = 10
        batch_concatenate_clips(video_paths, batch_size, output_path)

        shutil.rmtree("videos")
    else:
        logging.error("No subtitle data found or ASL data is missing.")
