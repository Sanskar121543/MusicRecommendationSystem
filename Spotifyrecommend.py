import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import datetime

import numpy as np
import pandas as pd
import requests
import sounddevice as sd
from requests.exceptions import RequestException, Timeout, SSLError
from textblob import TextBlob

# Configure logging to file instead of console
log_file = 'mood_music.log'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file)]
)
logger = logging.getLogger(__name__)

class SpeechRecognizer:
    """Handles speech recognition using Vosk model."""
    
    def __init__(self, model_path: str = "vosk-model-small-en-us-0.15"):
        self.model_path = model_path
        self.recognizer = None
        
    def initialize(self):
        """Lazily initialize the speech recognition model."""
        try:
            from vosk import Model, KaldiRecognizer
            
            if not os.path.exists(self.model_path):
                logger.error(f"Model path '{self.model_path}' does not exist.")
                print(f"❌ Error: Speech recognition model not found at '{self.model_path}'.")
                print("   Please follow these steps:")
                print("   1. Download the model from https://alphacephei.com/vosk/models")
                print("   2. Extract the .tar.xz file using 7-Zip or similar")
                print("   3. Place the extracted folder in the same directory as this script")
                print("   4. Make sure the folder name is exactly: vosk-model-small-en-us-0.15")
                return False
                
            # Check if model files exist
            required_files = ['conf', 'am', 'graph', 'ivector']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(self.model_path, f))]
            if missing_files:
                logger.error(f"Missing required model files: {missing_files}")
                print(f"❌ Error: The model folder is missing required files: {', '.join(missing_files)}")
                print("   Please make sure you extracted the model correctly")
                return False
                
            logger.info(f"Loading Vosk model from {self.model_path}")
            model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(model, 16000)
            self.recognizer.SetWords(True)
            return True
        except ImportError:
            logger.error("Vosk library not installed")
            print("❌ Error: The Vosk library is not installed.")
            print("   Please install it with: pip install vosk")
            return False
        except Exception as e:
            logger.error(f"Error initializing speech recognizer: {e}")
            print(f"❌ Error initializing speech recognizer: {e}")
            return False
    
    def record_audio(self) -> Optional[bytes]:
        """Record audio from microphone and return audio data."""
        if not self.recognizer and not self.initialize():
            return None
            
        try:
            print("🎤 Recording... Press Enter to stop.")
            recorded_frames = []

            def callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Status during recording: {status}")
                recorded_frames.append(indata.copy())

            # Create a non-blocking input method
            with sd.InputStream(samplerate=16000, blocksize=8000, dtype="int16",
                               channels=1, callback=callback):
                try:
                    # Wait for user to press Enter to stop recording
                    input()  
                except KeyboardInterrupt:
                    pass
                    
            print("🛑 Recording stopped.")
            
            if not recorded_frames:
                logger.warning("No audio data recorded")
                return None
                
            audio_data = np.concatenate(recorded_frames, axis=0)
            return audio_data.tobytes()
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            print(f"❌ Error recording audio: {e}")
            return None
    
    def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio data to text."""
        if not self.recognizer and not self.initialize():
            return ""
            
        try:
            # Process audio in chunks
            chunk_size = 16000  # 1 second of audio at 16kHz
            chunks = [audio_data[i:i + chunk_size] for i in range(0, len(audio_data), chunk_size)]
            
            transcript = ""
            for chunk in chunks:
                if len(chunk) > 0:
                    if self.recognizer.AcceptWaveform(chunk):
                        result = json.loads(self.recognizer.Result())
                        transcript += " " + result.get("text", "")
            
            # Get final result
            final_result = json.loads(self.recognizer.FinalResult())
            transcript += " " + final_result.get("text", "")
            
            transcript = transcript.strip()
            
            if not transcript:
                logger.warning("No speech detected in the audio")
                print("⚠️ No speech detected in the audio.")
                
            return transcript
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            print(f"❌ Error transcribing audio: {e}")
            return ""


class GeoLocationService:
    """Handles geolocation and related services."""
    
    def __init__(self):
        self.location_api = "https://ipinfo.io/json"
        self.weather_api_base = "https://wttr.in"
        self.max_retries = 2
        self.retry_delay = 1  # seconds
        
    def get_geo_info(self) -> Dict[str, Any]:
        """Get user's geographical information."""
        try:
            response = requests.get(self.location_api, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # Extract location coordinates if available
            lat, lon = None, None
            if "loc" in data and "," in data["loc"]:
                lat, lon = data["loc"].split(",")
                
            return {
                "city": data.get("city", "Unknown"),
                "region": data.get("region", "Unknown"),
                "country": data.get("country", "Unknown"),
                "timezone": data.get("timezone"),
                "latitude": lat,
                "longitude": lon,
                "isp": data.get("org"),
            }
        except Timeout:
            logger.error("Timeout when fetching geo info")
            print("⚠️ Location service unavailable")
            return {"city": "Unknown", "region": "Unknown", "country": "Unknown"}
        except RequestException as e:
            logger.error(f"Error fetching geo info: {e}")
            print("⚠️ Location unavailable")
            return {"city": "Unknown", "region": "Unknown", "country": "Unknown"}
        except Exception as e:
            logger.error(f"Unexpected error in geo info: {e}")
            return {"city": "Unknown", "region": "Unknown", "country": "Unknown"}
            
    def get_weather(self, city: str) -> Tuple[Optional[str], Optional[str]]:
        """Get weather information for a city."""
        if not city or city == "Unknown":
            return None, None
            
        try:
            url = f"{self.weather_api_base}/{city}?format=%t+%C"
            
            for attempt in range(self.max_retries + 1):
                try:
                    response = requests.get(url, timeout=5)
                    response.raise_for_status()
                    
                    weather_data = response.text.strip().split(" ", 1)
                    temperature = weather_data[0] if len(weather_data) > 0 else None
                    condition = weather_data[1] if len(weather_data) > 1 else None
                    
                    return temperature, condition
                except (Timeout, SSLError, RequestException) as e:
                    if attempt < self.max_retries:
                        logger.warning(f"Retry {attempt+1}/{self.max_retries} for weather after error: {e}")
                        time.sleep(self.retry_delay)
                    else:
                        raise
                        
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return None, None
            
    def get_time_of_day(self, timezone: Optional[str]) -> Optional[str]:
        """Get time of day based on user's timezone."""
        if not timezone:
            return self._get_local_time_of_day()
            
        # Try multiple methods to get time of day
        for method in [self._get_time_from_api, self._get_time_from_timezone, self._get_local_time_of_day]:
            time_of_day = method(timezone)
            if time_of_day:
                return time_of_day
                
        return None
        
    def _get_time_from_api(self, timezone: str) -> Optional[str]:
        """Try to get time from worldtimeapi.org with retries."""
        url = f"https://worldtimeapi.org/api/timezone/{timezone}"
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(url, timeout=3)
                response.raise_for_status()
                
                data = response.json()
                dt = data.get("datetime")
                
                if dt and len(dt) >= 13:
                    hour = int(dt[11:13])
                    return self._hour_to_time_of_day(hour)
            except (SSLError, Timeout, RequestException) as e:
                if attempt < self.max_retries:
                    logger.warning(f"Retry {attempt+1}/{self.max_retries} for time API after error: {e}")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to get time from API: {e}")
                    break
                    
        return None
        
    def _get_time_from_timezone(self, timezone: str) -> Optional[str]:
        """Try to calculate time from timezone using Python's datetime."""
        try:
            import pytz
            now = datetime.datetime.now(pytz.timezone(timezone))
            return self._hour_to_time_of_day(now.hour)
        except Exception as e:
            logger.warning(f"Failed to get time from timezone: {e}")
            return None
            
    def _get_local_time_of_day(self, _=None) -> str:
        """Fallback to local system time."""
        now = datetime.datetime.now()
        return self._hour_to_time_of_day(now.hour)
        
    def _hour_to_time_of_day(self, hour: int) -> str:
        """Convert hour to time of day string."""
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"


class SentimentAnalyzer:
    """Analyzes text sentiment."""
    
    def analyze(self, text: str) -> str:
        """Analyze sentiment of the given text."""
        try:
            analysis = TextBlob(text)
            
            # Classify sentiment based on polarity
            if analysis.sentiment.polarity > 0.2:
                return "positive"
            elif analysis.sentiment.polarity < -0.2:
                return "negative"
            else:
                return "neutral"
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return "neutral"  # Default to neutral on error
            
    def map_to_mood(self, sentiment: str) -> str:
        """Map sentiment to a mood descriptor."""
        sentiment_to_mood = {
            "positive": "Happy",
            "negative": "Sad",
            "neutral": "Neutral"
        }
        return sentiment_to_mood.get(sentiment, "Neutral")


class MusicRecommender:
    """Recommends songs based on mood and context."""
    
    def __init__(self, csv_path: str = "high_popularity_spotify_data.csv"):
        self.csv_path = csv_path
        self.song_data = None
        
    def load_dataset(self) -> bool:
        """Load song dataset from CSV file."""
        try:
            if not os.path.exists(self.csv_path):
                logger.error(f"Song dataset not found: {self.csv_path}")
                print(f"❌ Error: Song dataset not found at '{self.csv_path}'")
                return False
                
            logger.info(f"Loading song dataset from {self.csv_path}")
            self.song_data = pd.read_csv(self.csv_path)
            
            # Validate required columns
            required_cols = ["track_name", "track_artist", "valence"]
            missing_cols = [col for col in required_cols if col not in self.song_data.columns]
            
            if missing_cols:
                logger.error(f"Dataset missing required columns: {missing_cols}")
                print(f"❌ Error: Dataset missing required columns: {missing_cols}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error loading song dataset: {e}")
            print(f"❌ Error loading song dataset: {e}")
            return False
    
    def recommend(self, mood: str, weather_condition: Optional[str], time_of_day: Optional[str]) -> List[str]:
        """Recommend songs based on mood, weather, and time of day."""
        if self.song_data is None and not self.load_dataset():
            return ["Could not load song dataset."]
            
        try:
            # Start with a copy of all songs
            filtered_songs = self.song_data.copy()
            
            # Filter by mood (using valence as primary indicator)
            if mood == "Happy":
                filtered_songs = filtered_songs[filtered_songs["valence"] > 0.6]
            elif mood == "Sad":
                filtered_songs = filtered_songs[filtered_songs["valence"] < 0.4]
            else:  # Neutral
                filtered_songs = filtered_songs[(filtered_songs["valence"] >= 0.4) & (filtered_songs["valence"] <= 0.6)]
                
            # Further refine by weather if available
            if weather_condition and len(filtered_songs) > 10:
                weather_keywords = {
                    "rain": ["Rain", "Storm", "Drizzle", "Umbrella", "Thunder"],
                    "sun": ["Sun", "Bright", "Shine", "Summer", "Heat"],
                    "cloud": ["Cloud", "Grey", "Gloomy", "Fog", "Mist"],
                    "snow": ["Snow", "Winter", "Cold", "Ice", "Freeze"],
                    "wind": ["Wind", "Breeze", "Hurricane", "Tornado", "Gale"]
                }
                
                # Find matching weather category
                weather_lower = weather_condition.lower()
                for category, keywords in weather_keywords.items():
                    if any(keyword.lower() in weather_lower for keyword in [category] + keywords):
                        # Search for these keywords in track names
                        weather_songs = filtered_songs[
                            filtered_songs["track_name"].str.contains('|'.join(keywords), case=False, na=False)
                        ]
                        
                        # If we found at least 3 songs, use this subset
                        if len(weather_songs) >= 3:
                            filtered_songs = weather_songs
                            break
            
            # Further refine by time of day if available (using a softer approach)
            if time_of_day and len(filtered_songs) > 10:
                time_keywords = {
                    "Morning": ["Morning", "Dawn", "Sunrise", "Early", "Wake"],
                    "Afternoon": ["Afternoon", "Day", "Lunch", "Noon", "Sun"],
                    "Evening": ["Evening", "Sunset", "Dusk", "Twilight"],
                    "Night": ["Night", "Dark", "Star", "Moon", "Dream", "Sleep"]
                }
                
                if time_of_day in time_keywords:
                    keywords = time_keywords[time_of_day]
                    time_songs = filtered_songs[
                        filtered_songs["track_name"].str.contains('|'.join(keywords), case=False, na=False)
                    ]
                    
                    # Only use this subset if we found enough songs
                    if len(time_songs) >= 3:
                        filtered_songs = time_songs
            
            # If we don't have enough songs after all filtering, revert to mood-only filter
            if len(filtered_songs) < 3:
                if mood == "Happy":
                    filtered_songs = self.song_data[self.song_data["valence"] > 0.6]
                elif mood == "Sad":
                    filtered_songs = self.song_data[self.song_data["valence"] < 0.4]
                else:  # Neutral
                    filtered_songs = self.song_data[(self.song_data["valence"] >= 0.4) & (self.song_data["valence"] <= 0.6)]
            
            # Sample 5 songs (or fewer if we don't have 5)
            num_recommendations = min(5, len(filtered_songs))
            if num_recommendations == 0:
                return ["No suitable songs found in the dataset."]
                
            recommended_songs = filtered_songs.sample(num_recommendations)
            
            # Format recommendations
            result = []
            for _, row in recommended_songs.iterrows():
                song = f"🎵 {row['track_name']} by {row['track_artist']}"
                # Add valence indicator if available
                if 'valence' in row and not pd.isna(row['valence']):
                    mood_level = row['valence']
                    mood_indicator = "😊" if mood_level > 0.7 else "🙂" if mood_level > 0.5 else "😐" if mood_level > 0.3 else "☹️"
                    song += f" {mood_indicator}"
                result.append(song)
            
            return result
            
        except Exception as e:
            logger.error(f"Error recommending songs: {e}")
            return [f"Error recommending songs: {e}"]


class SentimentAudioDataset:
    """Handles the speech sentiment dataset for testing."""
    
    def __init__(self, csv_path: str = "speech_sentiment_dataset.csv"):
        self.csv_path = csv_path
        self.dataset = None
        
    def load_dataset(self) -> bool:
        """Load the speech sentiment dataset."""
        try:
            if not os.path.exists(self.csv_path):
                logger.error(f"Speech dataset not found: {self.csv_path}")
                return False
                
            self.dataset = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.dataset)} speech samples")
            return True
        except Exception as e:
            logger.error(f"Error loading speech dataset: {e}")
            return False
            
    def get_sample(self, index: int) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Get a sample from the dataset by index."""
        if self.dataset is None and not self.load_dataset():
            return None, None, None
            
        if index < 0 or index >= len(self.dataset):
            return None, None, None
            
        row = self.dataset.iloc[index]
        return row.get("filename"), row.get("transcript"), row.get("sentiment")


class MoodMusicApp:
    """Main application class."""
    
    def __init__(self):
        self.speech_recognizer = SpeechRecognizer()
        self.geo_service = GeoLocationService()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.music_recommender = MusicRecommender()
        self.test_dataset = SentimentAudioDataset()
        
    def run(self):
        """Run the main application."""
        self.print_header()
        
        while True:
            choice = self.get_input_mode()
            
            if choice == "q":
                print("👋 Goodbye!")
                break
                
            transcript = ""
            
            # Handle different input modes
            if choice == "1":  # Audio input
                print("Please wait, initializing speech recognition...")
                audio_data = self.speech_recognizer.record_audio()
                if audio_data:
                    transcript = self.speech_recognizer.transcribe(audio_data)
            elif choice == "2":  # Text input
                transcript = input("Please enter your text: ").strip()
            elif choice == "3":  # Test dataset
                self.run_test_mode()
                continue
            else:
                print("❌ Invalid choice.")
                continue
                
            if not transcript:
                print("❌ No input text to analyze.")
                continue
                
            self.process_and_recommend(transcript)
            
            # Ask if the user wants to continue
            if not self.continue_prompt():
                print("👋 Goodbye!")
                break
    
    def print_header(self):
        """Print the application header."""
        print("\n" + "=" * 60)
        print("🎵 🎭  MOOD MUSIC RECOMMENDATION SYSTEM  🎭 🎵")
        print("=" * 60)
        print("This system recommends songs based on your mood,")
        print("the weather in your location, and the time of day.")
        print("=" * 60 + "\n")
    
    def get_input_mode(self) -> str:
        """Get the user's preferred input mode."""
        print("\nSelect input type:")
        print("1) 🎤 Audio Input (speak your mood)")
        print("2) ⌨️  Text Input (type your mood)")
        print("q) 🚪 Quit")
        return input("Enter your choice (1/2/q): ").strip().lower()
    
    def run_test_mode(self):
        """Run the application in test mode using the dataset."""
        if not self.test_dataset.load_dataset():
            print("❌ Could not load test dataset.")
            return
            
        print("\n== 🧪 TEST MODE ==")
        print(f"Found {len(self.test_dataset.dataset)} test samples.")
        
        while True:
            try:
                idx_input = input("\nEnter sample index (0-4) or 'q' to return to main menu: ").strip().lower()
                if idx_input == 'q':
                    break
                    
                idx = int(idx_input)
                filename, transcript, sentiment = self.test_dataset.get_sample(idx)
                
                if not transcript:
                    print(f"❌ Invalid sample index: {idx}")
                    continue
                    
                print(f"\n🔍 Using test sample {idx}:")
                print(f"📝 Transcript: \"{transcript}\"")
                print(f"✅ Original sentiment: {sentiment}")
                
                # Process the transcript as if it was user input
                self.process_and_recommend(transcript)
                
            except ValueError:
                print("❌ Please enter a valid number or 'q'.")
            except Exception as e:
                print(f"❌ Error in test mode: {e}")
    
    def process_and_recommend(self, transcript: str):
        """Process input text and generate recommendations."""
        print("\n== 🔍 ANALYSIS RESULTS ==")
        print(f"📝 Input: \"{transcript}\"")
        
        # Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze(transcript)
        mood = self.sentiment_analyzer.map_to_mood(sentiment)
        print(f"📊 Sentiment: {sentiment} (Mood: {mood})")
        
        # Get geographic info and context
        geo_info = self.geo_service.get_geo_info()
        print(f"🌍 Location: {geo_info.get('city')}, {geo_info.get('region')}, {geo_info.get('country')}")
        
        # Get weather
        temperature, weather_condition = self.geo_service.get_weather(geo_info.get('city'))
        if temperature and weather_condition:
            print(f"🌡️ Weather: {temperature}, {weather_condition}")
        else:
            print("🌡️ Weather: Unknown")
            weather_condition = None
        
        # Get time of day
        time_of_day = self.geo_service.get_time_of_day(geo_info.get('timezone'))
        if time_of_day:
            print(f"⏰ Time: {time_of_day}")
        else:
            print("⏰ Time: Unknown")
        
        print("\n== 🎵 SONG RECOMMENDATIONS ==")
        print(f"Finding songs for your {mood.lower()} mood...")
        
        # Get recommendations
        recommendations = self.music_recommender.recommend(mood, weather_condition, time_of_day)
        
        # Display recommendations
        for recommendation in recommendations:
            print(recommendation)
            
        print("\n" + "-" * 60)
    
    def continue_prompt(self) -> bool:
        """Ask the user if they want to continue."""
        response = input("\nWould you like another recommendation? (y/n): ").strip().lower()
        return response.startswith('y')


def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    required_packages = ["numpy", "pandas", "requests", "sounddevice", "textblob", "vosk"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Some required packages are missing:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
        
    return True


if __name__ == "__main__":
    try:
        # Check dependencies before starting
        if not check_dependencies():
            sys.exit(1)
            
        # Create and run the application
        app = MoodMusicApp()
        app.run()
    except KeyboardInterrupt:
        print("\n\n👋 Program terminated by user. Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ An unexpected error occurred: {e}")