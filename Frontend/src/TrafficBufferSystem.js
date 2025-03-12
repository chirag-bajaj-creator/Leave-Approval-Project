import React, { useState } from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// ✅ Define Custom Marker Icon
const customMarker = new L.Icon({
  iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
  shadowSize: [41, 41],
});

export default function TrafficBufferSystem() {
  const [empId, setEmpId] = useState("");
  const [employeeLocation, setEmployeeLocation] = useState(null);

  // Fixed Office Location (Delhi Office)
  const officeLocation = { lat: 28.6139, lon: 77.2090 };

  // Fetch Employee Location from Backend
  const fetchEmployeeLocation = async () => {
    try {
      const response = await fetch(`http://localhost:5000/get-employee-location?empId=${empId}`);
      const data = await response.json();

      if (data && data.lat && data.lon) {
        setEmployeeLocation({ lat: data.lat, lon: data.lon });
      } else {
        alert("Employee not found!");
      }
    } catch (error) {
      console.error("Error fetching employee location:", error);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center space-y-6">
      <h2 className="text-2xl font-semibold text-gray-800">🚦 Employee Traffic & Buffer Time</h2>

      {/* Employee ID Input */}
      <div className="flex space-x-4">
        <input
          type="text"
          placeholder="Enter Employee ID"
          className="px-4 py-2 border border-gray-300 rounded-lg"
          value={empId}
          onChange={(e) => setEmpId(e.target.value)}
        />
        <button
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
          onClick={fetchEmployeeLocation}
        >
          Get Location
        </button>
      </div>

      {/* Map Container */}
      <MapContainer
        center={employeeLocation ? [employeeLocation.lat, employeeLocation.lon] : [officeLocation.lat, officeLocation.lon]}
        zoom={5}
        className="h-[350px] w-full rounded-lg shadow-md"
      >
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

        {/* ✅ Show Only One Marker at a Time */}
        {employeeLocation ? (
          <Marker position={[employeeLocation.lat, employeeLocation.lon]} icon={customMarker}>
            <Popup>📍 Employee Location</Popup>
          </Marker>
        ) : (
          <Marker position={[officeLocation.lat, officeLocation.lon]} icon={customMarker}>
            <Popup>🏢 Office Location</Popup>
          </Marker>
        )}
      </MapContainer>
    </div>
  );
}
