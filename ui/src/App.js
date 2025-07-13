// src/App.js
import React, { useState, useEffect, useRef } from 'react';
import FileUpload from './components/FileUpload';
import ChatInterface from './components/ChatInterface';
import AudioOutput from './components/AudioOutput';
import { uploadAudio, sendMessage, resetSession as apiResetSession, getUserSessions, getSessionHistory } from './api'; // <-- NEW IMPORTS
import './App.css';

import { auth, googleProvider } from './firebaseConfig';
import {
    onAuthStateChanged,
    signInWithPopup,
    signOut
} from 'firebase/auth';

const API_URL = "http://localhost:8000";

function App() {
    const [sessionId, setSessionId] = useState(null);
    const [fileUploaded, setFileUploaded] = useState(false);
    const [currentFilename, setCurrentFilename] = useState(null);
    const [initialAudioUrl, setInitialAudioUrl] = useState(null);
    const [chatHistory, setChatHistory] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isDarkTheme, setIsDarkTheme] = useState(true);
    const chatContainerRef = useRef(null);

    const [user, setUser] = useState(null);
    const [authError, setAuthError] = useState(null);

    const [userSessions, setUserSessions] = useState([]);

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

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
            setUser(currentUser);
            if (currentUser) {
                console.log("User logged in:", currentUser.displayName || currentUser.email || "via Google", "UID:", currentUser.uid);
                fetchUserSessions(currentUser.uid);
                if (!sessionId) { // Only generate a new session ID if one isn't already active
                    setSessionId(uuidv4());
                }
            } else {
                console.log("User logged out");
                setSessionId(null);
                setFileUploaded(false);
                setCurrentFilename(null);
                setInitialAudioUrl(null);
                setChatHistory([]);
                setUserSessions([]);
            }
        });

        return () => unsubscribe();
    }, []);

    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [chatHistory]);


    const uuidv4 = () => {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            var r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    };

    const fetchUserSessions = async (userId) => {
        if (!userId) return;
        setIsProcessing(true); // Indicate loading
        try {
            const sessions = await getUserSessions(userId);
            setUserSessions(sessions);
        } catch (error) {
            console.error("Error fetching user sessions:", error);
            setAuthError("Failed to load your chat sessions."); // Or a more specific error message
        } finally {
            setIsProcessing(false);
        }
    };

    const handleSessionClick = async (sessionIdToLoad) => {
        if (!user?.uid || sessionIdToLoad === sessionId) { // Prevent re-loading same session
            return;
        }

        setIsProcessing(true);
        try {
            const sessionData = await getSessionHistory(sessionIdToLoad, user.uid);

            setSessionId(sessionIdToLoad);
            const history = (sessionData.messages || []).map(item => {
                const files = item.files;
                const stems = files.filter(file => file.file_type === 'stem').map(item => {
                    item.name = item.stem;
                    return item;
                });
                const remix = files.find(file => file.file_type === 'remix');
                item.stems = stems;
                item.remix = remix;

                return item;
            })
            console.log(history);
            setChatHistory(history);
            setFileUploaded(true); // Check if files exist for this session

            setInitialAudioUrl(`${API_URL}${sessionData.audio_path}`);
            setCurrentFilename(sessionData.audio_path.replace('\/downloads\/', ''));

            console.log(`Loaded history for session: ${sessionIdToLoad}`);
            setTimeout(() => {
                if (chatContainerRef.current) {
                    chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
                }
            }, 0);
        } catch (error) {
            console.error(`Error fetching history for session ${sessionIdToLoad}:`, error);
            setAuthError("Failed to load chat history. It might be corrupted or not belong to you.");
        } finally {
            setIsProcessing(false);
        }
    };

    const resetApplicationState = async () => {
        if (user && sessionId) {
            try {
                await apiResetSession(sessionId, user.uid); // Pass user.uid
                console.log(`Backend session ${sessionId} for user ${user.uid} deleted.`);
            } catch (error) {
                console.error("Error resetting backend session:", error);
                setAuthError("Failed to delete old session."); // Inform user
            }
        }

        if (initialAudioUrl) {
            URL.revokeObjectURL(initialAudioUrl);
            console.log("Revoked local audio URL:", initialAudioUrl);
        }

        const newSessionId = user ? uuidv4() : null;
        setSessionId(newSessionId);
        setFileUploaded(false);
        setCurrentFilename(null);
        setInitialAudioUrl(null);
        setChatHistory([]);
        setIsProcessing(false);
        setAuthError(null); // Clear any previous auth errors

        if (user) {
            await fetchUserSessions(user.uid);
        }
        console.log("Application state reset. New session ID:", newSessionId);
    };

    const handleFileUpload = async (file) => {
        if (!file || !user || !sessionId) { // Ensure user, and sessionId are available
            console.error("File, user, or session ID missing for upload.");
            setChatHistory([{ role: 'assistant', content: '❌ Please log in and ensure a session is active to upload files.' }]);
            return;
        }

        setIsProcessing(true);
        setCurrentFilename(file.name);
        setChatHistory([]); // Clear chat history for new upload context
        setAuthError(null);

        const localAudioUrl = URL.createObjectURL(file);
        setInitialAudioUrl(localAudioUrl);

        try {
            const response = await uploadAudio(file, sessionId, user.uid); // Pass user.uid
            if (response.success) {
                setFileUploaded(true);
                setChatHistory([{ role: 'assistant', content: 'File ready! You can now start chatting below.' }]);
                await fetchUserSessions(user.uid);
            } else {
                console.error("Upload failed:", response.error);
                setFileUploaded(false);
                setCurrentFilename(null);
                if (localAudioUrl) URL.revokeObjectURL(localAudioUrl);
                setInitialAudioUrl(null);
                setChatHistory([{ role: 'assistant', content: `❌ Upload failed: ${response.error}` }]);
            }
        } catch (error) {
            console.error("Connection Error during upload:", error);
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
        if (!message.trim() || !user || !sessionId || !fileUploaded) { // Ensure user, session and file are ready
            console.error("Message, user, session ID, or file not uploaded for sending message.");
            return;
        }

        const newUserMessage = { role: 'user', content: message, timestamp: new Date().toISOString() };
        setChatHistory(prev => [...prev, newUserMessage]);
        setIsProcessing(true);
        setAuthError(null);

        try {
            const response = await sendMessage(sessionId, message, user.uid); // Pass user.uid
            if (response.success) {
                const rawHistory = response.data.history || [];
                const stems = response.data.stems || [];
                const remix = response.data.remix || null;

                const lastAssistantMsgIndex = rawHistory.map(m => m.role).lastIndexOf('assistant');

                if (lastAssistantMsgIndex > -1 && (stems.length > 0 || remix)) {
                    rawHistory[lastAssistantMsgIndex].stems = stems;
                    rawHistory[lastAssistantMsgIndex].remix = remix;
                }

                setChatHistory(prev => [...prev, rawHistory[lastAssistantMsgIndex]])
                // const updatedHistory = rawHistory.map(msg => ({
                //     ...msg,
                //     timestamp: new Date(msg.timestamp)
                // }));

                // setChatHistory(updatedHistory);

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

    const handleGoogleLogin = async () => {
        setAuthError(null);
        try {
            await signInWithPopup(auth, googleProvider);
        } catch (error) {
            setAuthError(error.message);
            console.error("Google Login Error:", error);
        }
    };

    const handleLogout = async () => {
        try {
            await signOut(auth);
        } catch (error) {
            console.error("Logout Error:", error);
        }
    };

    if (!user) {
        return (
            <div className={`auth-container ${isDarkTheme ? 'dark-theme' : 'light-theme'}`}>
                <h1>SoundScribe - Login</h1>
                <div className="auth-form google-only-form">
                    <p>Please sign in to continue.</p>
                    {authError && <p className="auth-error">{authError}</p>}
                    <button onClick={handleGoogleLogin} className="google-login-button">
                        <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" alt="Google logo" />
                        Sign in with Google
                    </button>
                </div>

                <button
                    onClick={() => setIsDarkTheme(prev => !prev)}
                    className="theme-toggle-button"
                    title={isDarkTheme ? "Switch to Light Theme" : "Switch to Dark Theme"}
                >
                    {isDarkTheme ? '☀️' : '🌙'}
                </button>
            </div>
        );
    }

    return (
        <div className="app-layout">
            <aside className="sidebar">
                <div className="sidebar-header">
                    <button className="new-chat-button" onClick={resetApplicationState} disabled={isProcessing}>
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
                    <h3>Your Conversations</h3>
                    {isProcessing && userSessions.length === 0 ? (
                        <p>Loading chats...</p>
                    ) : userSessions.length === 0 ? (
                        <p className="no-conversations-msg">No conversations yet. Click '+ New Chat' to start one!</p>
                    ) : (
                        <ul className="session-list">
                            {userSessions.map((session) => (
                                <li
                                    key={session.id}
                                    className={`session-item ${session.id === sessionId ? 'active' : ''}`}
                                    onClick={() => handleSessionClick(session.id)}
                                >
                                    <span className="session-date">
                                        {new Date(session.created_at).toLocaleString('en-GB', {
                                            day: '2-digit', month: '2-digit', year: 'numeric',
                                            hour: '2-digit', minute: '2-digit'
                                        })}
                                    </span>
                                    <br />
                                    <small className="session-id-preview">
                                        ID: {session.id.substring(0, 8)}...
                                    </small>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>
                <div className="user-info">
                    {user && <p>Logged in as: <strong>{user.displayName || user.email || "Google User"}</strong></p>}
                    <button onClick={handleLogout} className="logout-button">
                        Logout
                    </button>
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

                            {/* ORIGINAL AUDIO */}
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

                            {/* CHAT INTERFACE */}
                            <ChatInterface
                                chatHistory={chatHistory}
                                onSendMessage={handleSendMessage}
                                isProcessing={isProcessing}
                                chatContainerRef={chatContainerRef}
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