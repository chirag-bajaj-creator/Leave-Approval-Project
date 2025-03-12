import React, { useState, useEffect } from "react";
import { io } from "socket.io-client";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import EmployeeDetails from "./components/EmployeeDetails";
import FileUpload from "./components/FileUpload";
import { motion, AnimatePresence } from "framer-motion";
import "leaflet/dist/leaflet.css";

const socket = io("http://localhost:3000");

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("leave");
  const [employeeLocations, setEmployeeLocations] = useState([
    { id: 1, name: "Amit", lat: 28.7041, lon: 77.1025 },
    // { id: 2, name: "Priya", lat: 19.0760, lon: 72.8777 },
    // { id: 3, name: "Rahul", lat: 12.9716, lon: 77.5946 },
  ]);
  const [trafficData, setTrafficData] = useState([]);

  useEffect(() => {
    fetchTrafficData();
  }, []);

  const fetchTrafficData = async () => {
    try {
      const response = await fetch(
        `https://router.project-osrm.org/route/v1/driving/77.1025,28.7041;72.8777,19.0760?overview=full`
      );
      const data = await response.json();
      if (data.routes) {
        setTrafficData(data.routes[0]);
      }
    } catch (error) {
      console.error("Error fetching traffic data:", error);
    }
  };

  // Function to render active tab content
  const renderActiveTab = () => {
    if (activeTab === "leave") {
      return (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-2xl font-semibold text-gray-800">
            📝 Leave Prediction System
          </h2>
          <EmployeeDetails />
        </motion.div>
      );
    } else if (activeTab === "traffic") {
      return (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-2xl font-semibold text-gray-800">
            🚦 Employee Traffic & Buffer Time
          </h2>

          {/* Traffic Map Section */}
          <MapContainer
            center={[20.5937, 78.9629]}
            zoom={5}
            className="h-[350px] w-full rounded-lg shadow-md"
          >
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
            {employeeLocations.map((emp) => (
              <Marker key={emp.id} position={[emp.lat, emp.lon]}>
                <Popup>{emp.name}'s Location</Popup>
              </Marker>
            ))}
          </MapContainer>

          {/* Traffic Buffer Time Calculation */}
          <div className="p-4 bg-gray-100 rounded-lg shadow-md mt-4">
            <h3 className="text-lg font-semibold text-gray-700">
              ⏳ Buffer Time Calculation
            </h3>
            <p className="text-gray-600">
              <strong>Route Distance:</strong>{" "}
              {trafficData.distance
                ? (trafficData.distance / 1000).toFixed(2)
                : "Loading..."}{" "}
              km
            </p>
            <p className="text-gray-600">
              <strong>Estimated Travel Time:</strong>{" "}
              {trafficData.duration
                ? Math.round(trafficData.duration / 60)
                : "Loading..."}{" "}
              min
            </p>
            <p className="text-green-600 font-semibold">
              ✅ Buffer Time Granted:{" "}
              {trafficData.duration
                ? Math.ceil(trafficData.duration / 300) * 5
                : "Calculating..."}{" "}
              min
            </p>
          </div>
        </motion.div>
      );
    } else if (activeTab === "training") {
      return (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-2xl font-semibold text-gray-800">
            ⚡ Real-Time Model Training
          </h2>
          <FileUpload socket={socket} />
        </motion.div>
      );
    }
  };

  return (
    <div className="min-h-screen w-full bg-white flex">
      {/* Sidebar - Fixed on the Left */}
      <aside className="w-72 bg-white shadow-md fixed left-0 top-0 h-full flex flex-col p-6 border-r border-gray-200">
        <h1 className="text-2xl font-bold text-gray-800 mb-6">Dashboard</h1>
        <nav className="flex flex-col space-y-4">
          <button
            className={`w-full text-left px-4 py-3 rounded-lg transition-all duration-300 flex items-center space-x-2 text-lg font-medium ${
              activeTab === "leave"
                ? "bg-blue-500 text-white shadow-md"
                : "bg-gray-100 text-gray-800 hover:bg-gray-200"
            }`}
            onClick={() => setActiveTab("leave")}
          >
            <span>📝</span> <span>Leave Prediction</span>
          </button>

          <button
            className={`w-full text-left px-4 py-3 rounded-lg transition-all duration-300 flex items-center space-x-2 text-lg font-medium ${
              activeTab === "training"
                ? "bg-blue-500 text-white shadow-md"
                : "bg-gray-100 text-gray-800 hover:bg-gray-200"
            }`}
            onClick={() => setActiveTab("training")}
          >
            <span>⚡</span> <span>Model Training</span>
          </button>

          <button
            className={`w-full text-left px-4 py-3 rounded-lg transition-all duration-300 flex items-center space-x-2 text-lg font-medium ${
              activeTab === "traffic"
                ? "bg-green-500 text-white shadow-md"
                : "bg-gray-100 text-gray-800 hover:bg-gray-200"
            }`}
            onClick={() => setActiveTab("traffic")}
          >
            <span>🚦</span> <span>Traffic Buffer Time</span>
          </button>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex justify-center items-center ml-72 p-8">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.5 }}
          className="w-[800px] min-h-[500px] bg-white p-8 rounded-2xl shadow-lg border border-gray-200 transition-all duration-300"
        >
          <AnimatePresence mode="wait">{renderActiveTab()}</AnimatePresence>
        </motion.div>
      </main>
    </div>
  );
}