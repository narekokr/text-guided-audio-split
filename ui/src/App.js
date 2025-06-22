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
    const [initialAudioUrl, setInitialAudioUrl] = useState(null);
    const [chatHistory, setChatHistory] = useState([]);
    const [lastStems, setLastStems] = useState([]);
    const [lastRemix, setLastRemix] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isDarkTheme, setIsDarkTheme] = useState(true);
    const chatContainerRef = useRef(null);

    // Apply dark/light theme class to body
    useEffect(() => {
        if (isDarkTheme) {
            document.body.classList.add('dark-theme');
            document.body.classList.remove('light-theme');
        } else {
            document.body.classList.add('light-theme');
            document.body.classList.remove('dark-theme');
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

        if (initialAudioUrl) {
            URL.revokeObjectURL(initialAudioUrl);
            console.log("Revoked local audio URL:", initialAudioUrl);
        }

        setSessionId(uuidv4());
        setFileUploaded(false);
        setCurrentFilename(null);
        setInitialAudioUrl(null);
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
        setLastStems([]);
        setLastRemix(null);

        const localAudioUrl = URL.createObjectURL(file);
        setInitialAudioUrl(localAudioUrl);

        try {
            await apiResetSession(sessionId);

            const response = await uploadAudio(file, sessionId);
            if (response.success) {
                setFileUploaded(true);
                setChatHistory([{ role: 'assistant', content: 'File ready! You can now start chatting below.' }]);
            } else {
                console.error("Upload failed:", response.error);
                setFileUploaded(false);
                setCurrentFilename(null);
                if (localAudioUrl) URL.revokeObjectURL(localAudioUrl);
                setInitialAudioUrl(null);
                setChatHistory([{ role: 'assistant', content: `❌ Upload failed: ${response.error}` }]);
            }
        } catch (error) {
            console.error("Connection Error:", error);
            setFileUploaded(false);
            setCurrentFilename(null);
            if (localAudioUrl) URL.revokeObjectURL(localAudioUrl);
            setInitialAudioUrl(null);
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
        <div className="app-layout">
            <aside className="sidebar">
                <div className="sidebar-header">
                    <button className="new-chat-button" onClick={resetApplicationState}>
                        + New Chat
                    </button>
                    <button
                        onClick={() => setIsDarkTheme(prev => !prev)}
                        className="theme-toggle-button"
                        title={isDarkTheme ? "Switch to Light Theme" : "Switch to Dark Theme"}
                    >
                        {isDarkTheme ? '☀️' : '🌙'}
                    </button>
                </div>
                <div className="conversations-history">
                    {/* Conversation history will go here later */}
                    <p>No conversations yet.</p>
                </div>
            </aside>

            <main className="main-content">
                <div className="app-container">
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



                    {fileUploaded && (
                        <>
                            <hr className="divider" />
                            {!chatHistory.length && !isProcessing && (
                                <p className="info-message">Your audio is ready. Ask the assistant something, like 'separate vocals and bass'.</p>
                            )}

                            {/* FIRST: Original Audio */}
                            {initialAudioUrl && (
                                <>
                                    <h3>Original Audio</h3>
                                    <div className="output-container">
                                        <div className="audio-item">
                                            <strong>Original File</strong>
                                            <audio controls src={initialAudioUrl} className="audio-player"></audio>
                                            <a
                                                href={initialAudioUrl}
                                                download
                                                className="download-btn"
                                            >
                                                Download Original
                                            </a>
                                        </div>
                                    </div>
                                    <hr className="divider" />
                                </>
                            )}

                            {/* SECOND: Chat Interface */}
                            <ChatInterface
                                chatHistory={chatHistory}
                                onSendMessage={handleSendMessage}
                                isProcessing={isProcessing}
                                chatContainerRef={chatContainerRef}
                            />

                            {/* Only show divider if there will be generated audios and if chat history has content */}
                            {(lastStems.length > 0 || lastRemix) && (chatHistory.length > 0) && (
                                <hr className="divider" />
                            )}
                            <AudioOutput
                                lastStems={lastStems}
                                lastRemix={lastRemix}
                                apiBaseUrl={API_URL}
                            />
                        </>
                    )}
                </div>
            </main>
        </div>
    );
}

export default App;