import React, { useState } from "react";
import { io } from "socket.io-client";
import FileUpload from "../components/FileUpload";
import LeavePrediction from "../pages/LeavePrediction";
import TrafficBufferSystem from "../components/TrafficBufferSystem";

const socket = io("http://localhost:3000");

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("leave");

  return (
    <div className="min-h-screen w-full bg-white flex">
      {/* Sidebar - Fixed on the Left */}
      <aside className="w-80 bg-white shadow-md fixed left-0 top-0 h-full flex flex-col p-6 border-r border-gray-200">
        <h1 className="text-2xl font-bold text-gray-800 mb-6">Dashboard</h1>

        {/* Navigation Tabs */}
        <nav className="flex flex-col space-y-4">
          <button
            className={`w-full text-left px-4 py-3 rounded-lg transition-all duration-300 flex items-center space-x-2 text-lg font-medium ${
              activeTab === "leave"
                ? "bg-blue-500 text-white shadow-md"
                : "bg-gray-100 text-gray-800 hover:bg-gray-200"
            }`}
            onClick={() => setActiveTab("leave")}
          >
            📝 <span>Leave Prediction</span>
          </button>

          <button
            className={`w-full text-left px-4 py-3 rounded-lg transition-all duration-300 flex items-center space-x-2 text-lg font-medium ${
              activeTab === "traning"
                ? "bg-blue-500 text-white shadow-md"
                : "bg-gray-100 text-gray-800 hover:bg-gray-200"
            }`}
            onClick={() => setActiveTab("leave")}
          >
            📝 <span>Model Traning</span>
          </button>


          <button
            className={`w-full text-left px-4 py-3 rounded-lg transition-all duration-300 flex items-center space-x-2 text-lg font-medium ${
              activeTab === "traffic"
                ? "bg-green-500 text-white shadow-md"
                : "bg-gray-100 text-gray-800 hover:bg-gray-200"
            }`}
            onClick={() => setActiveTab("traffic")}
          >
            🚦 <span>Traffic Buffer Time</span>
          </button>
        </nav>

        {/* Divider */}
        <div className="border-t border-gray-300 my-6"></div>

        {/* ✅ Real-Time Model Training Section in Sidebar */}
        <h2 className="text-lg font-semibold text-gray-800 mb-3">⚡ Real-Time Model Training</h2>
        <FileUpload socket={socket} />
      </aside>

      {/* Main Content Box */}
      <main className="flex-1 flex justify-center items-center ml-80 p-8">
      if(activeTab === "leave"){
        <LeavePrediction />
        }elseif(activeTab === "traffic"){
        <TrafficBufferSystem />
        }else{
          <FileUpload/>
        }
        
      </main>
    </div>
  );
}
