import React from "react";

export function Button({ children, onClick }) {
  return (
    <button
      onClick={onClick}
      className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600 transition"
    >
      {children}
    </button>
  );
}