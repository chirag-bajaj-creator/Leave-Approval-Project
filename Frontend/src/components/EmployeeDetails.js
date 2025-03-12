import React, { useState } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { motion } from "framer-motion";

export default function EmployeeDetails() {
  const [employeeId, setEmployeeId] = useState("");
  const [employeeData, setEmployeeData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const fetchEmployeeDetails = async () => {
    if (!employeeId) {
      setErrorMessage("⚠️ Please enter an Employee ID!");
      return;
    }

    setLoading(true);
    setErrorMessage(""); // Clear previous errors

    try {
      const response = await fetch(`http://localhost:3000/employee-details/${employeeId}`);

      if (!response.ok) {
        throw new Error(`❌ Error: ${response.status} ${response.statusText}`);
      }

      const responseData = await response.json();
      console.log("API Response:", responseData); // Debugging

      setEmployeeData(responseData.data); // Store only the `data` part
    } catch (error) {
      console.error("Error fetching employee details:", error);
      setErrorMessage(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      className="p-6 bg-white/10 backdrop-blur-md rounded-lg shadow-lg border border-white/20 w-full max-w-lg mx-auto"
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h2 className="text-xl font-semibold text-white text-center mb-4">🔍 Employee Details</h2>

      <Label className="text-white mb-2">Employee ID</Label>
      <Input
        type="text"
        placeholder="Enter Employee ID"
        value={employeeId}
        onChange={(e) => setEmployeeId(e.target.value)}
        className="mb-3"
      />
      <Button onClick={fetchEmployeeDetails} className="w-full">Get Employee Details</Button>

      {loading && <p className="text-blue-400 mt-4 text-center">🔄 Fetching data...</p>}
      {errorMessage && <p className="text-red-500 mt-4 text-center">{errorMessage}</p>}

      {employeeData && (
        <div className="mt-6 p-4 bg-gray-100 text-black rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2 text-gray-900">📄 Employee Information</h3>
          <table className="w-full border-collapse border border-gray-300">
            <tbody>
              {Object.entries(employeeData).map(([key, value]) => (
                <tr key={key} className="border-b border-gray-300">
                  <td className="px-3 py-2 font-semibold text-gray-700">{key.replace(/_/g, " ")}:</td>
                  <td className="px-3 py-2 text-gray-600">{Array.isArray(value) ? value.join(", ") : value || "N/A"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </motion.div>
  );
}
