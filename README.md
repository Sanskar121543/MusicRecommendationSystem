# Mood Music Recommender

A web application that recommends music based on your mood, current weather, and time of day. The application supports both text and voice input for mood analysis.

## Features

- Text and voice input for mood analysis
- Sentiment analysis using TextBlob
- Location-based weather information
- Time-of-day aware recommendations
- Modern, responsive UI
- Real-time audio recording and processing

## Prerequisites

- Python 3.7 or higher
- Vosk speech recognition model
- Web browser with microphone access

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mood-music-recommender
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Vosk model:
- Visit https://alphacephei.com/vosk/models
- Download the `vosk-model-small-en-us-0.15` model
- Extract the model files to the project directory
- Ensure the model folder is named exactly `vosk-model-small-en-us-0.15`

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Choose your input method:
   - Text: Type how you're feeling
   - Voice: Click the record button and speak

4. View your personalized music recommendations based on:
   - Your current mood
   - Local weather conditions
   - Time of day

## Project Structure

```
mood-music-recommender/
├── app.py                 # Flask application
├── Spotifyrecommend.py   # Core recommendation logic
├── static/
│   ├── style.css         # Styles
│   └── script.js         # Frontend logic
├── templates/
│   └── index.html        # Main template
├── vosk-model-small-en-us-0.15/  # Speech recognition model
└── requirements.txt      # Python dependencies
```

## Notes

- The application requires microphone access for voice input
- Weather information is based on your IP location
- Song recommendations are from a curated dataset
- The Vosk model must be present in the project directory

## License

This project is licensed under the MIT License - see the LICENSE file for details. 