// api.js
const API_URL = process.env.REACT_APP_FIREBASE_API_URL;

export const uploadAudio = async (file, sessionId, userId) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', sessionId);
    formData.append('user_id', userId);

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

export const sendMessage = async (sessionId, message, userId) => {
    if (!userId) {
        return { success: false, error: "User ID is required for sending message." };
    }

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ session_id: sessionId, message, user_id: userId }),
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

export const resetSession = async (sessionId, userId) => {
    try {
        const response = await fetch(`${API_URL}/reset`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ session_id: sessionId, user_id: userId }),
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

export const getUserSessions = async (userId) => {
  const response = await fetch(`${API_URL}/user/${userId}/sessions`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
};

export const getSessionHistory = async (sessionId, userId) => {
  const response = await fetch(
    `${API_URL}/session/${sessionId}/history?user_id=${userId}`
  );
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
};