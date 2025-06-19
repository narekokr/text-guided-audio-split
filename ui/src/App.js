// src/App.js
import React, { useState, useEffect, useRef } from 'react';
import FileUpload from './components/FileUpload';
import ChatInterface from './components/ChatInterface';
import AudioOutput from './components/AudioOutput';
import { uploadAudio, sendMessage, resetSession as apiResetSession } from './api';
import './App.css';

const API_URL = "http://localhost:8000";

function App() {
    const [sessionId, setSessionId] = useState(null);
    const [fileUploaded, setFileUploaded] = useState(false);
    const [currentFilename, setCurrentFilename] = useState(null);
    const [chatHistory, setChatHistory] = useState([]);
    const [lastStems, setLastStems] = useState([]);
    const [lastRemix, setLastRemix] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isDarkTheme, setIsDarkTheme] = useState(true); // Start with dark theme
    const chatContainerRef = useRef(null);

    // Apply dark/light theme class to body
    useEffect(() => {
        if (isDarkTheme) {
            document.body.classList.add('dark-theme');
            document.body.classList.remove('light-theme'); // Ensure light theme is removed
        } else {
            document.body.classList.add('light-theme');
            document.body.classList.remove('dark-theme'); // Ensure dark theme is removed
        }
    }, [isDarkTheme]);

    // Initialize session ID on first load
    useEffect(() => {
        if (!sessionId) {
            setSessionId(uuidv4());
        }
    }, [sessionId]);

    const uuidv4 = () => {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            var r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    };

    const resetApplicationState = async () => {
        if (sessionId) {
            try {
                await apiResetSession(sessionId);
                console.log("Backend session reset successfully.");
            } catch (error) {
                console.error("Error resetting backend session:", error);
            }
        }
        setSessionId(uuidv4());
        setFileUploaded(false);
        setCurrentFilename(null);
        setChatHistory([]);
        setLastStems([]);
        setLastRemix(null);
        setIsProcessing(false);
    };

    const handleFileUpload = async (file) => {
        if (!file || !sessionId) return;

        setIsProcessing(true);
        setCurrentFilename(file.name);
        setChatHistory([]);

        try {
            await apiResetSession(sessionId);

            const response = await uploadAudio(file, sessionId);
            if (response.success) {
                setFileUploaded(true);
                setChatHistory([{ role: 'assistant', content: '✅ File ready! You can now start chatting below.' }]);
            } else {
                console.error("Upload failed:", response.error);
                setFileUploaded(false);
                setCurrentFilename(null);
                setChatHistory([{ role: 'assistant', content: `❌ Upload failed: ${response.error}` }]);
            }
        } catch (error) {
            console.error("Connection Error:", error);
            setFileUploaded(false);
            setCurrentFilename(null);
            setChatHistory([{ role: 'assistant', content: '❌ Connection Error. Is the API server running?' }]);
        } finally {
            setIsProcessing(false);
        }
    };

    const handleSendMessage = async (message) => {
        if (!message.trim() || !sessionId || !fileUploaded) return;

        const newUserMessage = { role: 'user', content: message };
        setChatHistory(prev => [...prev, newUserMessage]);
        setIsProcessing(true);
        setLastStems([]);
        setLastRemix(null);

        try {
            const response = await sendMessage(sessionId, message);
            if (response.success) {
                setChatHistory(response.data.history || []);
                setLastStems(response.data.stems || []);
                setLastRemix(response.data.remix || null);
            } else {
                console.error("Assistant failed:", response.error);
                setChatHistory(prev => [...prev, { role: 'assistant', content: `❌ Assistant failed: ${response.error}` }]);
            }
        } catch (error) {
            console.error("Connection error with assistant:", error);
            setChatHistory(prev => [...prev, { role: 'assistant', content: '❌ Connection Error. Could not contact the assistant.' }]);
        } finally {
            setIsProcessing(false);
        }
    };

    // Scroll chat to bottom on new messages
    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [chatHistory]);

    return (
        <div className="app-container">
            <div className="theme-toggle-container">
                <button
                    onClick={() => setIsDarkTheme(prev => !prev)}
                    className="theme-toggle-button"
                    title={isDarkTheme ? "Switch to Light Theme" : "Switch to Dark Theme"}
                >
                    {isDarkTheme ? '☀️' : '🌙'}
                </button>
            </div>

            <h1>SoundScribe - Audio Assistant</h1>
            {currentFilename ? (
                <p className="caption"><strong>Now chatting about:</strong> <code>{currentFilename}</code></p>
            ) : (
                <p className="caption">Upload an audio file below to get started.</p>
            )}

            <FileUpload
                onFileUpload={handleFileUpload}
                isProcessing={isProcessing}
                fileUploaded={fileUploaded}
                resetApplicationState={resetApplicationState}
            />

            {(fileUploaded || chatHistory.length > 0) && (
                <button onClick={resetApplicationState} className="reset-button">
                    🔄 Start New Session
                </button>
            )}

            {fileUploaded && (
                <>
                    <hr />
                    {!chatHistory.length && !isProcessing && (
                        <p className="info-message">Your audio is ready. Ask the assistant something, like 'separate vocals and bass'.</p>
                    )}

                    <ChatInterface
                        chatHistory={chatHistory}
                        onSendMessage={handleSendMessage}
                        isProcessing={isProcessing}
                        chatContainerRef={chatContainerRef}
                    />

                    <AudioOutput
                        lastStems={lastStems}
                        lastRemix={lastRemix}
                        apiBaseUrl={API_URL}
                    />
                </>
            )}
        </div>
    );
}

export default App;