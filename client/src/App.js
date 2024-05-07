import React, { useState, useEffect } from 'react'
import VideoInput from "./VideoInput";
import VideoDisplay from './VideoDisplay';
import "./styles.css";

const VideoUploader = () => {
  const [videoUrl, setVideoUrl] = useState('');

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
  };

  return (
    <div>
      <h1>Video Uploader</h1>
      <input type="file" onChange={handleFileChange} accept="video/*" />
      {videoUrl && (
        <video controls>
          <source src={videoUrl} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      )}
    </div>
  );
};



function App() {
  const backendVideoUrl = 'file:///C:/Users/adney/Videos/Captures/Blade-of-Honor%20-%20BossFight%20-%20WebGL%20-%20Unity%202022.3.15f1_%20_DX11_%202024-05-03%2009-08-57.mp4';

  return (
    <div>
      <div className='container_title'>
        <h1 className='title'>traffic-sign-ar</h1>
      </div>

      <div className="container_videos">
      <div className="square"><VideoInput  /></div>
      <div className="square"><VideoDisplay videoUrl={backendVideoUrl} /></div>
      </div>
    </div>
  )
}

export default App

