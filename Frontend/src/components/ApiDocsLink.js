import React from "react";

export default function ApiDocsLink() {
  return (
    <div className="text-center mt-6">
      <a
        href="http://localhost:3000/apidocs/"
        target="_blank"
        className="text-blue-400 hover:text-blue-500 transition text-lg"
      >
        📄 View API Documentation
      </a>
    </div>
  );
}
