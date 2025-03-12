import React from "react";

export function Progress({ value }) {
  return (
    <div className="w-full bg-gray-200 h-4 rounded">
      <div className="bg-green-500 h-4 rounded" style={{ width: `${value}%` }}></div>
    </div>
  );
}
