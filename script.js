document.addEventListener('DOMContentLoaded', () => {
    // Tab switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            
            // Update active tab button
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update active tab content
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${tab}-tab`) {
                    content.classList.add('active');
                }
            });
        });
    });

    // Text analysis
    const textInput = document.getElementById('text-input');
    const analyzeTextBtn = document.getElementById('analyze-text');

    analyzeTextBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) {
            showError('Please enter some text to analyze');
            return;
        }

        showLoading();
        try {
            const response = await fetch('/analyze_text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            const data = await response.json();
            if (response.ok) {
                displayResults(data);
            } else {
                showError(data.error || 'Failed to analyze text');
            }
        } catch (error) {
            console.error('Text analysis error:', error);
            showError('An error occurred while analyzing text');
        } finally {
            hideLoading();
        }
    });

    // Audio recording
    const recordBtn = document.getElementById('record-btn');
    const recordingStatus = document.getElementById('recording-status');
    let audioContext = null;
    let audioStream = null;
    let audioProcessor = null;
    let audioData = [];
    let isRecording = false;

    recordBtn.addEventListener('click', async () => {
        if (isRecording) {
            // Stop recording
            isRecording = false;
            recordBtn.classList.remove('recording');
            recordingStatus.textContent = 'Processing...';
            
            // Stop all tracks and processors
            if (audioStream) {
                audioStream.getTracks().forEach(track => track.stop());
            }
            if (audioProcessor) {
                audioProcessor.disconnect();
            }
            if (audioContext) {
                await audioContext.close();
            }

            // Process and send audio data
            try {
                if (audioData.length === 0) {
                    showError('No audio data was recorded. Please try again.');
                    recordingStatus.textContent = '';
                    return;
                }

                // Combine all audio chunks
                const totalLength = audioData.reduce((acc, chunk) => acc + chunk.length, 0);
                const combinedData = new Int16Array(totalLength);
                let offset = 0;
                for (const chunk of audioData) {
                    combinedData.set(chunk, offset);
                    offset += chunk.length;
                }

                // Convert to blob and send
                const audioBlob = new Blob([combinedData.buffer], { type: 'audio/raw' });
                const formData = new FormData();
                formData.append('audio', audioBlob);

                showLoading();
                const response = await fetch('/analyze_audio', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                if (response.ok) {
                    displayResults(data);
                } else {
                    showError(data.error || 'Failed to analyze audio');
                }
            } catch (error) {
                console.error('Error processing audio:', error);
                showError('An error occurred while processing audio: ' + error.message);
            } finally {
                hideLoading();
                recordingStatus.textContent = '';
                audioData = []; // Clear audio data
            }
            return;
        }

        try {
            // Reset audio data
            audioData = [];
            isRecording = true;

            // Request audio with specific constraints
            audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });

            // Create audio context
            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });

            // Create audio source
            const source = audioContext.createMediaStreamSource(audioStream);

            // Create script processor for raw audio data
            audioProcessor = audioContext.createScriptProcessor(4096, 1, 1);

            // Process audio data
            audioProcessor.onaudioprocess = (e) => {
                if (!isRecording) return;
                
                const inputData = e.inputBuffer.getChannelData(0);
                // Convert float32 to int16
                const pcmData = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    pcmData[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                }
                audioData.push(pcmData);
            };

            // Connect the nodes
            source.connect(audioProcessor);
            audioProcessor.connect(audioContext.destination);

            // Start recording
            recordBtn.classList.add('recording');
            recordingStatus.textContent = 'Recording... Click again to stop';
        } catch (error) {
            console.error('Error accessing microphone:', error);
            showError('Failed to access microphone. Please ensure microphone permissions are granted.');
            isRecording = false;
        }
    });

    // Display results
    function displayResults(data) {
        const resultsSection = document.getElementById('results');
        const loadingElement = document.getElementById('loading');
        const errorElement = document.getElementById('error');
        const transcriptText = document.getElementById('transcript-text');
        
        // Hide loading and error elements
        loadingElement.classList.add('hidden');
        errorElement.classList.add('hidden');
        
        // Display transcription if available
        if (data.transcription) {
            transcriptText.textContent = data.transcription;
            document.getElementById('transcript-display').classList.remove('hidden');
        } else {
            document.getElementById('transcript-display').classList.add('hidden');
        }
        
        // Display mood information
        const moodDisplay = document.getElementById('mood-display');
        const locationDisplay = document.getElementById('location-display');
        const weatherDisplay = document.getElementById('weather-display');
        const timeDisplay = document.getElementById('time-display');
        
        // Update mood info
        moodDisplay.textContent = data.mood || 'Unknown';
        locationDisplay.textContent = data.location || 'Unknown';
        weatherDisplay.textContent = data.weather || 'Unknown';
        timeDisplay.textContent = data.time_of_day || 'Unknown';
        
        // Display recommendations
        const recommendationsList = document.getElementById('recommendations-list');
        recommendationsList.innerHTML = '';
        
        if (data.recommendations && data.recommendations.length > 0) {
            data.recommendations.forEach(recommendation => {
                const item = document.createElement('div');
                item.className = 'recommendation-item';
                item.innerHTML = `
                    <span class="song-emoji">🎵</span>
                    <span class="song-info">${recommendation}</span>
                `;
                recommendationsList.appendChild(item);
            });
        } else {
            recommendationsList.innerHTML = '<p>No recommendations available</p>';
        }
        
        // Show results section
        resultsSection.classList.remove('hidden');
    }

    // Utility functions
    function showLoading() {
        document.getElementById('loading').classList.remove('hidden');
        document.getElementById('results').classList.add('hidden');
    }

    function hideLoading() {
        document.getElementById('loading').classList.add('hidden');
    }

    function showError(message) {
        const errorDiv = document.getElementById('error');
        errorDiv.querySelector('p').textContent = message;
        errorDiv.classList.remove('hidden');
        document.getElementById('results').classList.add('hidden');
    }
}); 