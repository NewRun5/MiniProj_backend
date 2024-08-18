import React, { useState, useEffect } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [labelFilter, setLabelFilter] = useState([]);
  const [detResult, setDetResult] = useState(null);
  const [filteredLabels, setFilteredLabels] = useState([]);
  const [videoUrl, setVideoUrl] = useState(null);
  const [timeline, setTimeline] = useState([]); // 타임라인 데이터를 상태로 관리
  const [filteredTimeline, setFilteredTimeline] = useState([]); // 필터링된 타임라인

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleLabelFilterChange = (e) => {
    const labels = e.target.value.split(',');
    setLabelFilter(labels);
    filterTimeline(labels); // 레이블 변경 시 필터링된 타임라인 업데이트
  };

  const filterTimeline = (labels) => {
    if (timeline.length === 0) return;

    const filtered = timeline.filter(event => labels.includes(event.label));
    setFilteredTimeline(filtered);
  };

  const uploadFile = async () => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://localhost:8000/uploadfile/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setDetResult(response.data.det_result);
      setFilteredLabels(response.data.label_list);
      setTimeline(response.data.timeline); // 타임라인 데이터 설정
      filterTimeline(labelFilter); // 초기 타임라인 필터링
    } catch (error) {
      console.error("Error uploading file:", error);
    }
  };

  const processAndPlayVideo = async () => {
    if (!detResult || !labelFilter.length) {
      alert("필터링할 레이블과 결과를 선택하세요.");
      return;
    }

    const formData = new FormData();
    formData.append("label_filter", JSON.stringify(labelFilter));
    formData.append("det_result", JSON.stringify(detResult));
    formData.append("file", file);

    try {
      const response = await axios.post("http://localhost:8000/result-video/", formData, {
        responseType: "blob",
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      setVideoUrl(url);

    } catch (error) {
      console.error("Error processing video:", error);
    }
  };

  const downloadVideo = () => {
    if (!videoUrl) {
      alert("먼저 비디오를 생성하세요.");
      return;
    }

    const link = document.createElement("a");
    link.href = videoUrl;
    link.setAttribute("download", "result_video.mp4");
    document.body.appendChild(link);
    link.click();
  };

  useEffect(() => {
    filterTimeline(labelFilter); // labelFilter가 변경될 때마다 타임라인 필터링
  }, [labelFilter, timeline]);

  return (
    <div className="App">
      <h1>Object Detection Video Processor</h1>

      <input type="file" accept="video/*" onChange={handleFileChange} />
      <button onClick={uploadFile}>Upload and Process Video</button>

      {filteredLabels.length > 0 && (
        <div>
          <h3>Select Labels to Filter</h3>
          <input
            type="text"
            value={labelFilter.join(",")}
            onChange={handleLabelFilterChange}
            placeholder="e.g., person,car,dog"
          />
          <button onClick={processAndPlayVideo} disabled={!detResult}>
            Generate and Play Result Video
          </button>
        </div>
      )}

      {videoUrl && (
        <div style={{ display: "flex", flexDirection: "row" }}>
          <div>
            <h3>Processed Video</h3>
            <video controls src={videoUrl} style={{ width: "100%", maxWidth: "600px" }}></video>
            <button onClick={downloadVideo}>Download Result Video</button>
          </div>
          <div style={{ marginLeft: "20px" }}>
            <h3>Filtered Timeline</h3>
            <ul>
              {filteredTimeline.map((event, index) => (
                <li key={index}>
                  ID: {event.id}, Label: {event.label}, Start: {event.start}s, End: {event.end}s
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
