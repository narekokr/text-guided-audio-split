// api.js
const API_URL = "http://localhost:8000";

export const uploadAudio = async (file, sessionId) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);

    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();
        if (response.ok) {
            return { success: true, data };
        } else {
            return { success: false, error: data.detail || 'Upload failed' };
        }
    } catch (error) {
        console.error("API Error during upload:", error);
        return { success: false, error: 'Network error or API is down' };
    }
};

export const sendMessage = async (sessionId, message) => {
    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ session_id: sessionId, message }),
        });
        const data = await response.json();
        if (response.ok) {
            return { success: true, data };
        } else {
            return { success: false, error: data.detail || 'Assistant failed to respond' };
        }
    } catch (error) {
        console.error("API Error during chat:", error);
        return { success: false, error: 'Network error or API is down' };
    }
};

export const resetSession = async (sessionId) => {
    try {
        const response = await fetch(`${API_URL}/reset`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ session_id: sessionId }),
        });
        const data = await response.json();
        if (response.ok) {
            return { success: true, data };
        } else {
            console.error("Failed to reset backend session:", data.detail || 'Unknown error');
            return { success: false, error: data.detail || 'Failed to reset backend session' };
        }
    } catch (error) {
        console.error("Network error during session reset:", error);
        return { success: false, error: 'Network error during session reset' };
    }
};
