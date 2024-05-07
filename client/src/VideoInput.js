import React from "react";

export default function VideoInput(props) {
  const { width, height } = props;

  const inputRef = React.useRef();
  const [source, setSource] = React.useState();

  const handleFileChange = async (event) => {
    const file = event.target.files[0];

    // Form that will send the uploaded video to the backend
    const formData = new FormData();
    formData.append("video", file);
    try {
      const response = await fetch("http://localhost:5000/video-input", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to upload video");
      }

      const data = await response.json();
      // Handle successful upload response (e.g., display success message)
      console.log("Video upload successful:", data);
    } catch (error) {
      console.error("Error uploading video:", error);
    }

    const url = URL.createObjectURL(file);
    setSource(url);
  };

  const handleChoose = (event) => {
    inputRef.current.click();
  };

  return (
    <div className="VideoInput">
      <input
        ref={inputRef}
        className="VideoInput_input"
        type="file"
        onChange={handleFileChange}
        accept=".mov,.mp4"
      />
      {!source && <button onClick={handleChoose}>Choose</button>}
      {source && (
        <video
          className="VideoInput_video"
          width="100%"
          height={height}
          controls
          src={source}
        />
      )}
      <div className="VideoInput_footer">{source || "Nothing selectd"}</div>
    </div>
  );
}
