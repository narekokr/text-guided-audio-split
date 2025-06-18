// components/FileUpload.js
import React, { useState, useRef } from 'react';
import './FileUpload.css'; // For styling

function FileUpload({ onFileUpload, isProcessing, fileUploaded }) {
    const [selectedFile, setSelectedFile] = useState(null);
    const fileInputRef = useRef(null);

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const handleSubmit = (event) => {
        event.preventDefault();
        if (selectedFile) {
            onFileUpload(selectedFile);
        }
    };

    const triggerFileInput = () => {
        fileInputRef.current.click();
    };

    return (
        <div className={`file-upload-container ${fileUploaded ? 'collapsed' : ''}`}>
            <details open={!fileUploaded}> {/* Use details/summary for expander effect */}
                <summary className="expander-summary">Upload a New Audio File</summary>
                <form onSubmit={handleSubmit}>
                    <input
                        type="file"
                        accept=".mp3,.wav,.m4a"
                        onChange={handleFileChange}
                        disabled={isProcessing}
                        style={{ display: 'none' }} // Hide default input
                        ref={fileInputRef}
                    />
                    <button
                        type="button" // Important: use type="button" to prevent form submission
                        onClick={triggerFileInput}
                        disabled={isProcessing}
                        className="custom-file-input-button"
                    >
                        {selectedFile ? selectedFile.name : "Choose File"}
                    </button>
                    {selectedFile && <span className="selected-filename">{selectedFile.name}</span>}
                    <button type="submit" disabled={!selectedFile || isProcessing}>
                        {isProcessing ? 'Processing...' : 'Process Audio'}
                    </button>
                </form>
            </details>
        </div>
    );
}

export default FileUpload;
