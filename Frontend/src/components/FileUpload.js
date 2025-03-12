import React, { useState, useEffect } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Progress } from "./ui/progress";
import { motion } from "framer-motion";

export default function FileUpload({ socket }) {
  const [file, setFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [predictionData, setPredictionData] = useState(null);
  const [statusMessage, setStatusMessage] = useState("Waiting for input...");

  useEffect(() => {
    if (!socket) {
      console.error("Socket is undefined! Check if it's being passed properly.");
      return;
    }

    // Listening for training progress from server
    socket.on("progress", (data) => {
      if (data.status.includes("Training Started")) {
        setStatusMessage("📤 Model Training Started");
      } else {
        setStatusMessage(data.status);
      }
      
      setUploadProgress((prev) => (prev < 100 ? prev + 20 : 100));

      if (data.data) {
        setPredictionData(data.data);
      }
    });

    return () => {
      if (socket) {
        socket.off("progress");
      }
    };
  }, [socket]);

  const handleFileUpload = async () => {
    if (!file) {
      setStatusMessage("❌ No file selected!");
      return;
    }

    setUploadProgress(10);
    setStatusMessage("📤 Uploading file...");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:3000/model-training", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setStatusMessage("📤 Model Training Started"); // Set status when model starts training
        setPredictionData(data.download);
        setUploadProgress(100);
        setStatusMessage("✅ Model trained successfully!");
      } else {
        setStatusMessage(`❌ Error: ${data.error}`);
      }
    } catch (error) {
      setStatusMessage(`❌ Upload failed: ${error.message}`);
    }
  };

  return (
    <motion.div
      className="p-4 bg-white rounded-lg shadow-lg border border-gray-300"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Label className="text-black font-medium">Upload CSV File</Label>
      <Input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])} className="mt-2" />
      <Button onClick={handleFileUpload} className="mt-2 bg-blue-600 text-white hover:bg-blue-700">Upload & Train Model</Button>
      <Progress value={uploadProgress} className="mt-2" />
      <p className="text-sm mt-2 text-black">{statusMessage}</p>

      {predictionData && (
        <div className="mt-4 p-2 border rounded bg-gray-100 text-black">
          <h3 className="font-semibold">Prediction Results:</h3>
          <pre className="text-black">{JSON.stringify(predictionData, null, 2)}</pre>
        </div>
      )}
    </motion.div>
  );
}
