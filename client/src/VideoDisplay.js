import React, { useState, useEffect } from 'react';

const VideoDisplay = ({ videoUrl }) => {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Simulate fetching video URL from backend (replace with your actual logic)
        const response = await fetch(videoUrl);

        if (!response.ok) {
          throw new Error('Failed to fetch video');
        }

        const videoData = await response.blob();
        const url = URL.createObjectURL(videoData);
        setIsLoading(false);
        return url;
      } catch (error) {
        setError(error.message);
        setIsLoading(false);
      }
    };

    fetchData().then((url) => {
      if (url) {
        // Update video source dynamically if successful
        document.getElementById('video').src = url;
      }
    });
  }, [videoUrl]);

  if (isLoading) {
    return <p>Loading video...</p>;
  }

  if (error) {
    return <p>Error: {error}</p>;
  }

  return (
    <div>
      <video id="video" width="640" height="480" controls>
        <source src="" type="video/mp4" />
        Your browser does not support the video tag.
      </video>
    </div>
  );
};

export default VideoDisplay;
