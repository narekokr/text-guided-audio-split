// components/ChatInterface.js
import React from 'react';
import AudioOutput from './AudioOutput'; // Assuming AudioOutput is in the same directory
import './ChatInterface.css'; // For styling

function ChatInterface({ chatHistory, onSendMessage, isProcessing, chatContainerRef, apiBaseUrl }) {
    const [message, setMessage] = React.useState('');

    const handleSubmit = (event) => {
        event.preventDefault();
        if (message.trim()) {
            onSendMessage(message);
            setMessage('');
        }
    };

    return (
        <div className="chat-interface">
        <div className="chat-container" ref={chatContainerRef}>
                        {chatHistory.map((msg, index) => (
                            <div key={index} className={`chat-message-wrapper ${msg.role}-wrapper`}>

                                {msg.role === 'user' ? (
                                    <div className="chat-message user-msg">
                                        {msg.content}
                                    </div>
                                ) : (
                                    <div className="assistant-content-container">
                                        {msg.content && <div className="assistant-text">{msg.content}</div>}
                                        {(msg.stems?.length > 0 || msg.remix) && (
                                            <AudioOutput
                                                stems={msg.stems || []}
                                                remix={msg.remix || null}
                                                apiBaseUrl={apiBaseUrl}
                                            />
                                        )}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>

            <form onSubmit={handleSubmit} className="chat-form">
                <input
                    type="text"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="e.g., Isolate the drums"
                    disabled={isProcessing}
                />
                <button type="submit" disabled={isProcessing}>
                    {isProcessing ? 'Sending...' : 'Send'}
                </button>
            </form>
        </div>
    );
}

export default ChatInterface;