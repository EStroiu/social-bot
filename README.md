# social-bot
## Scripts

1. `robot_audio_code.py` - communicate with dialogflow on your desktop. Requires dialoglfow module running and .json service user key set up.
2. `robot_video_code.py` - detect faces with face-detection model on your desktop.
3. `combined_bot.py` - combination of `robot_audio_code.py` and `robot_video_code.py`, runs video and audio processing at the same time using threads (so not real multithreading yet). Requires both dialogflow and face-detection modules running.
4. `combined_bot_ollama.py` -  improved `combined_bot.py` that queries a local model through ollama. Will need to set up requests properly if not working on local network (maybe use a VPN).