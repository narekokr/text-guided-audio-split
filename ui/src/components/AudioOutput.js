// components/AudioOutput.js
import React from 'react';
import './AudioOutput.css'; // For styling

function AudioOutput({ lastStems, lastRemix, apiBaseUrl }) {
    return (
        <div className="audio-output-sections">
            {lastStems.length > 0 && (
                <>
                    <h3>🎧 Separated Stems</h3>
                    <div className="output-container">
                        {lastStems.map((stem, index) => (
                            <div key={index} className="audio-item">
                                <strong>{stem.name.charAt(0).toUpperCase() + stem.name.slice(1)}</strong>
                                <audio controls src={`${apiBaseUrl}${stem.file_url}`} className="audio-player"></audio>
                                <a
                                    href={`${apiBaseUrl}${stem.file_url}`}
                                    download={`${stem.name}.wav`}
                                    className="download-btn"
                                >
                                    ⬇️ Download {stem.name}.wav
                                </a>
                            </div>
                        ))}
                    </div>
                </>
            )}

            {lastRemix && (
                <>
                    <h3>🎛️ Remixed Audio</h3>
                    <div className="output-container">
                        <div className="audio-item">
                            <strong>{lastRemix.name ? lastRemix.name.charAt(0).toUpperCase() + lastRemix.name.slice(1) : "Remix"}</strong>
                            <audio controls src={`${apiBaseUrl}${lastRemix.file_url}`} className="audio-player"></audio>
                            <a
                                href={`${apiBaseUrl}${lastRemix.file_url}`}
                                download={`${lastRemix.name || "remix"}.wav`}
                                className="download-btn"
                            >
                                ⬇️ Download {lastRemix.name || "Remix"}.wav
                            </a>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}

export default AudioOutput;
