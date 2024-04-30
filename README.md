# Geo-Data
- script for a tryout: Geogusser- take a youtube link and a play along link, output the images ,location and comment



## Environment

- python=3.10

- pytorch=2.01,cuda=11.7

- others in requirements.txt(maybe you should install other tools,like flac,ffmpeg in your local environment)

##  Others

- To run the code, just :

  ~~~
  python script.py
  ~~~

  without any other options, because I hard-coded the YouTube link: https://www.youtube.com/watch?v=t98r-YV6LnQ  and the corresponding GeoGuessr link: https://www.geoguessr.com/challenge/j5QTVixXslrbDXHj into the code

- <strong style="color: red;">Warning, if you are not using a proxy, please delete the two lines of code at the beginning of utilities.py that set the proxy interface</strong>

- Cause not every youtube video has a subtitle file,so I have to covert the audio to text by myself,using [google speech to text api](https://cloud.google.com/speech-to-text/docs/quickstart),which costs a lot of time in running the program.  

- I use google map api to get the street view. **So you should have your own API key for it. You can looking for help [here](https://developers.google.com/streetview/publish)**

- **output in result.json**
