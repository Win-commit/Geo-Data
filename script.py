import requests
import json
from utilities import *


if __name__ == "__main__":


    GOOGLE_MAP_KEY="AIzaSyBXx-dirw_ai_W5xO9NSR80UtfPRLHmyo4"
    BASE_URL = "https://www.geoguessr.com/api/"
    change_token="j5QTVixXslrbDXHj"
    youtube_url="https://www.youtube.com/watch?v=t98r-YV6LnQ"  

    '''
    there may have other token to some other endpoints api,like game_token to v3/games/{game_token},
    trip_token to v3/trips/{trip_token}
    '''

    # Your token here,in the request header to profile\ endpoints
    _ncfa_TOKEN = "tcmnbM90xMj7IAWFDQjsN7XhSv5N1j8luOk8blcqYy4%3DZbb%2BDj5PfaVE%2B50KuOqC4XfifuXSgm4jkdIMQ8JY5JKEmjnPLOgOm0DeRrIveeL17XdFIX%2FQwufu32awQz5K5XHNVGg%2ByslFLRVyt0phzVE%3D" 

    # Create a session object and set the _ncfa cookie
    session = requests.Session()
    session.cookies.set("_ncfa", _ncfa_TOKEN, domain="www.geoguessr.com")

    # Set the User-Agent header
    session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    })

    base_info= session.post(BASE_URL + "v3/challenges/" + change_token,json={})


    if base_info.status_code != 200:
        print("Error:", base_info.status_code)
    else:
        video_path,audio_path=download_video(youtube_url)
        
        game_token = base_info.json()['token']
        locations = get_xy(session,game_token)
        locations_images=get_images(locations,"images",GOOGLE_MAP_KEY,"4096x2160","120")

        match_results=split_time(locations_images,video_path)

        texts=get_text(audio_path,match_results)

        for i in range(len(texts)):
            locations_images[i]["comments"]=texts[i]
        
        with open('result.json', 'w', encoding='utf-8') as f:
            json.dump(locations_images, f, ensure_ascii=False, indent=4)


    








