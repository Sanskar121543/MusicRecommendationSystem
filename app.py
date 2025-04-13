from flask import Flask, render_template, request, jsonify
import os
import tempfile
import wave
import io
import numpy as np
import traceback
from Spotifyrecommend import MoodMusicApp, SpeechRecognizer, SentimentAnalyzer, GeoLocationService, MusicRecommender

app = Flask(__name__)
mood_app = MoodMusicApp()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    text = request.json.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Get sentiment and mood
    sentiment = mood_app.sentiment_analyzer.analyze(text)
    mood = mood_app.sentiment_analyzer.map_to_mood(sentiment)
    
    # Get location and weather
    geo_info = mood_app.geo_service.get_geo_info()
    temperature, weather_condition = mood_app.geo_service.get_weather(geo_info.get('city'))
    time_of_day = mood_app.geo_service.get_time_of_day(geo_info.get('timezone'))
    
    # Get recommendations
    recommendations = mood_app.music_recommender.recommend(mood, weather_condition, time_of_day)
    
    return jsonify({
        'sentiment': sentiment,
        'mood': mood,
        'location': f"{geo_info.get('city', 'Unknown')}, {geo_info.get('region', 'Unknown')}, {geo_info.get('country', 'Unknown')}",
        'weather': f"{temperature or 'Unknown'}, {weather_condition or 'Unknown'}",
        'time_of_day': time_of_day or 'Unknown',
        'recommendations': recommendations
    })

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    try:
        # Read the raw audio data
        raw_audio = audio_file.read()
        
        if not raw_audio:
            return jsonify({'error': 'Empty audio data received'}), 400
        
        # Convert bytes to numpy array of int16
        try:
            audio_data = np.frombuffer(raw_audio, dtype=np.int16)
        except Exception as e:
            return jsonify({'error': f'Error converting audio data: {str(e)}'}), 400
        
        if len(audio_data) == 0:
            return jsonify({'error': 'No audio data detected'}), 400
        
        # Create a temporary WAV file
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                wav_path = temp_wav.name
                
                # Write WAV file with correct parameters
                with wave.open(wav_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
                    wav_file.setframerate(16000)  # 16kHz sample rate
                    wav_file.writeframes(audio_data.tobytes())
            
            # Initialize speech recognizer
            speech_recognizer = SpeechRecognizer()
            if not speech_recognizer.initialize():
                return jsonify({'error': 'Failed to initialize speech recognition'}), 500
            
            # Read the WAV file
            with open(wav_path, 'rb') as f:
                audio_data = f.read()
            
            # Transcribe audio
            transcript = speech_recognizer.transcribe(audio_data)
            
            if not transcript:
                return jsonify({'error': 'No speech detected in audio'}), 400
            
            # Process the transcript
            sentiment = mood_app.sentiment_analyzer.analyze(transcript)
            mood = mood_app.sentiment_analyzer.map_to_mood(sentiment)
            
            # Get location and weather
            geo_info = mood_app.geo_service.get_geo_info()
            temperature, weather_condition = mood_app.geo_service.get_weather(geo_info.get('city'))
            time_of_day = mood_app.geo_service.get_time_of_day(geo_info.get('timezone'))
            
            # Get recommendations
            recommendations = mood_app.music_recommender.recommend(mood, weather_condition, time_of_day)
            
            return jsonify({
                'transcription': transcript,
                'sentiment': sentiment,
                'mood': mood,
                'location': f"{geo_info.get('city', 'Unknown')}, {geo_info.get('region', 'Unknown')}, {geo_info.get('country', 'Unknown')}",
                'weather': f"{temperature or 'Unknown'}, {weather_condition or 'Unknown'}",
                'time_of_day': time_of_day or 'Unknown',
                'recommendations': recommendations
            })
            
        finally:
            # Clean up temporary file
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception as e:
                    print(f"Error cleaning up temporary file: {e}")
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Audio processing error: {error_details}")
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 