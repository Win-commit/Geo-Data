import requests
import os
from model import VideoImageComparator
import cv2
from pytube import YouTube
from pydub import AudioSegment
from google.cloud import speech
import wave
from tqdm import tqdm

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
def get_xy(session,game_token):
    '''
    some questions: in the game_info response,there is a key named state,and round,when the value of state is "finished",
    we can confirm we have got the all labels(locations).

    but sometimes the _ncfa_TOKEN owner don't finish the challenge,especially meeting the new challenge
    we need to decide when to use 'get' way or 'post' way to get the information we want

    However ,in this case, i have finished the challenge.
    '''
    locations = []

    game_info = session.get("https://www.geoguessr.com/api/" + "v3/games/" + game_token).json()
    
    if game_info['state'] == "started":
        # do the post action
        pass
    else:
        #finished the challenge
        for round in game_info['rounds']:
            locations.append([round['lat'],round['lng']])
            
    return locations


def get_images(locations:list,save_folder,GOOGLE_MAP_KEY,size,fov):
    '''
    documents: https://developers.google.com/maps/documentation/streetview/overview
    '''
    res_ls=[]
    for location in locations:

        lat=location[0]
        lng=location[1]
        # get four directions of one location
        headings=[0,90,180,270]
        imgs_path=[]

        for heading in headings:
            url = "https://maps.googleapis.com/maps/api/streetview?"+\
              "size="+size+\
              "&location="+str(lat)+","+str(lng)+\
              "&fov="+fov+\
              "&heading="+str(heading)+\
              "&key="+GOOGLE_MAP_KEY
            
            response = requests.get(url)

            if response.status_code == 200:
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                with open(f"{save_folder}/{lat}_{lng}_{heading}.jpg", "wb") as file:
                    file.write(response.content)
                    imgs_path.append(f"{save_folder}/{lat}_{lng}_{heading}.jpg")
            else:
                print("Error:", response.status_code,"for",location)

        res_ls.append({'location':location,'imgs_path':imgs_path})

    res_ls.append({'location':['NULL','NULL'],'imgs_path':["./end1.jpg","./end2.jpg"]})
    return res_ls




def find_consecutive_nums(numbers,target_num,num_of_sequences):
    sequences = []  # A list used to store the start positions and lengths
    current_start = None  # Used to record the starting position of the current sequence
    current_length = 0  # Used to record the length of the current sequence

    for i, number in enumerate(numbers):
        if number == target_num:
            if current_length == 0:
                current_start = i  
            current_length += 1  
        else:
            if current_length > 0:
                sequences.append((current_start, current_length)) 
                current_start = None  
                current_length = 0  


    if current_length > 0:
        sequences.append((current_start, current_length))


    longest_sequences = sorted(sequences, key=lambda x: x[1], reverse=True)[:num_of_sequences]
    res = sorted(longest_sequences, key=lambda x: x[0], reverse=False)
    return res


def split_time(data,video_file,skip_seconds=5):
    
    cap = cv2.VideoCapture(video_file)

    comparator = VideoImageComparator()

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(frame_rate * skip_seconds)
    success, frame = cap.read()
    frame_count = 0
    
    match_each_frame=[]
    #Using tqdm to show the progress
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames,desc="Process frame") as pbar:
        while success:
            res=[]
            for item in data:
                target_image_paths=item["imgs_path"]
                target_images = [cv2.imread(target_image_path) for target_image_path in target_image_paths]
                res.append(comparator.compare_images_ssim(frame, target_images))
            
            match_each_frame.append(res.index(min(res)))

        
            frame_count += frame_skip
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            success, frame = cap.read()
            # update progress bar
            pbar.update(frame_skip)

    #TODO: find the start/end time of each picture.Specialiy ,find the len(data)-1 number appears most consecutively
    res = find_consecutive_nums(match_each_frame,len(data)-1,len(data))
    res=[(i[0]+i[1]-1)*skip_seconds for i in res]

    return res


def download_video(video_url,save_folder="video"):
    yt = YouTube(video_url)
    filepaths=[]
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Select the video stream of the highest quality
    video_stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if video_stream:
        # download video
        file_name= os.path.join(save_folder,"video_file.mp4")
        filepaths.append(file_name)
        video_stream.download(filename=file_name)

    #   Select the audio stream of the highest quality
    audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
    if audio_stream:
        # download audio
        file_name= os.path.join(save_folder,"audio_file.mp3")
        filepaths.append(file_name)
        audio_stream.download(filename=file_name)
        # convert mp3 to channel 1 wav
        audio = AudioSegment.from_file(filepaths.pop())
        #To adapt the api of google speech to text: 1 channel and 16 bit
        mono_audio = audio.set_channels(1)
        mono_audio = mono_audio.set_sample_width(2) 

        file_name= os.path.join(save_folder,"audio_mono.wav")
        filepaths.append(file_name)
        mono_audio.export(file_name, format='wav')
    
    return filepaths

def get_sample_rate(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        return sample_rate
    

def get_text(wav_path:str,timestampes:list):
    
    audio = AudioSegment.from_file(wav_path)
    smaple_rate=get_sample_rate(wav_path)
    audio_ls=[audio[timestampes[i]*1000:timestampes[i+1]*1000] for i in range(len(timestampes)-1)]
    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=smaple_rate,
        language_code="en-US",
    )

    all_text=[]

    for item in tqdm(audio_ls,desc="Processing audios", position=0):
        item_text=[]
        split_length = 20000
        audio_chunks=[item[i:i + split_length] for i in range(0, len(item), split_length)]

        for audio_chunk in tqdm(audio_chunks, desc="Chunks", leave=False, position=1):
            audio_sample = speech.RecognitionAudio(content=audio_chunk.raw_data)
            response = client.recognize(config=config, audio=audio_sample)
            for result in response.results:
                item_text.append(result.alternatives[0].transcript)
        all_text.append(item_text)
    
    return all_text



    

    
    



