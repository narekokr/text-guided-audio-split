import React from 'react';
import './AudioOutput.css';

function AudioOutput({ stems, remix, apiBaseUrl }) {
    const hasGeneratedAudio = stems.length > 0 || remix;

    if (!hasGeneratedAudio) {
        return null; // Don't render anything if there's no generated audio for the message
    }

    return (
        <div className="audio-output-sections">
            {stems.length > 0 && (
                <>
                    <h3>Separated Stems</h3>
                    <div className="output-container">
                        {stems.map((stem, index) => (
                            <div key={index} className="audio-item">
                                <strong>{stem.name.charAt(0).toUpperCase() + stem.name.slice(1)}</strong>
                                <audio controls src={`${apiBaseUrl}${stem.file_url}`} className="audio-player"></audio>
                                <a
                                    href={`${apiBaseUrl}${stem.file_url}`}
                                    download={`${stem.name}.wav`}
                                    className="download-btn"
                                >
                                    Download {stem.name}.wav
                                </a>
                            </div>
                        ))}
                    </div>
                </>
            )}

            {remix && (
                <>
                    <h3>Remixed Audio</h3>
                    <div className="output-container">
                        <div className="audio-item">
                            <strong>{remix.name ? remix.name.charAt(0).toUpperCase() + remix.name.slice(1) : "Remix"}</strong>
                            <audio controls src={`${apiBaseUrl}${remix.file_url}`} className="audio-player"></audio>
                            <a
                                href={`${apiBaseUrl}${remix.file_url}`}
                                download={`${remix.name || "remix"}.wav`}
                                className="download-btn"
                            >
                                Download {remix.name || "Remix"}.wav
                            </a>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}

export default AudioOutput;