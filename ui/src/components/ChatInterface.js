// components/ChatInterface.js
import React, { useState, useEffect, useRef } from 'react';
import './ChatInterface.css'; // For styling

function ChatInterface({ chatHistory, onSendMessage, isProcessing, chatContainerRef }) {
    const [message, setMessage] = useState('');

    const handleSubmit = (event) => {
        event.preventDefault();
        if (message.trim()) {
            onSendMessage(message);
            setMessage('');
        }
    };

    return (
        <div className="chat-interface">
            <hr />
            <div className="chat-container" ref={chatContainerRef}>
                {chatHistory.map((msg, index) => (
                    <div key={index} className={`chat-message ${msg.role}-msg`}>
                        {msg.content}
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
