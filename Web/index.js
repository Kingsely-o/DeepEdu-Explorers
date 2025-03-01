import React, { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Sidebar } from "@/components/ui/sidebar";
import { Switch } from "@/components/ui/switch";
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";

const logo = "https://github.com/Kingsely-o/waylo.github.io/blob/main/df_icon/icon.png?raw=true";

export default function App() {
  return (
    <Router>
      <div className="flex h-screen">
        <Sidebar />
        <div className="flex-1 p-6">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/deepfake-detection" element={<DeepShield />} />
            <Route path="/result-output" element={<ResultOutput />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

function Dashboard() {
  return (
    <div>
      <h2 className="text-2xl font-bold">Dashboard - Welcome to DeepShield</h2>
      <p className="text-lg text-gray-600 mt-2 mb-6 italic font-semibold">Exposing Deepfakes, Protecting Reality.</p>
      <div className="mt-8 grid grid-cols-2 gap-6 text-center">
        <div className="p-6 border rounded-lg shadow-lg transition-transform transform hover:scale-105">
          <h3 className="text-xl font-bold mb-4">Video Detection</h3>
          <Button className="bg-blue-500 text-white px-8 py-3 rounded-lg text-lg font-semibold hover:bg-blue-700 transition">Upload</Button>
        </div>
        <div className="p-6 border rounded-lg shadow-lg transition-transform transform hover:scale-105">
          <h3 className="text-xl font-bold mb-4">Audio Detection</h3>
          <Button className="bg-blue-500 text-white px-8 py-3 rounded-lg text-lg font-semibold hover:bg-blue-700 transition">Upload</Button>
        </div>
      </div>
    </div>
  );
}

function DeepShield() {
  return (
    <div>
      <h2 className="text-2xl font-bold">Deepfake Detection</h2>
      <p className="text-md text-gray-600 mb-4">Analyze and detect deepfake media content.</p>
      <div className="p-6 border rounded-lg shadow-lg text-center">
        <p className="text-gray-500">Deepfake analysis results will be displayed here.</p>
      </div>
    </div>
  );
}

function ResultOutput() {
  return (
    <div>
      <h2 className="text-2xl font-bold">Result Output</h2>
      <div className="grid grid-cols-2 gap-6 mt-6">
        <div className="p-4 border rounded-lg shadow-lg text-center">
          <h3 className="text-xl font-bold">Video 1 Output</h3>
          <p className="text-red-600 font-semibold text-lg">Result: Fake</p>
          <img src="https://github.com/Kingsely-o/waylo.github.io/blob/main/df_icon/V1.png?raw=true" alt="Video 1 Output" className="w-full rounded shadow-md mt-2" />
        </div>
        <div className="p-4 border rounded-lg shadow-lg text-center">
          <h3 className="text-xl font-bold">Video 2 Output</h3>
          <p className="text-green-600 font-semibold text-lg">Result: True</p>
          <img src="https://github.com/Kingsely-o/waylo.github.io/blob/main/df_icon/V2.png?raw=true" alt="Video 2 Output" className="w-full rounded shadow-md mt-2" />
        </div>
      </div>
    </div>
  );
}

function Sidebar() {
  return (
    <div className="w-64 p-4 bg-gray-100 h-full">
      <div className="flex items-center space-x-2">
        <img src={logo} alt="DeepShield Logo" className="w-10 h-10" />
        <h2 className="text-lg font-bold">DeepShield</h2>
      </div>
      <nav className="mt-4">
        <ul className="space-y-2">
          <li><Link to="/" className="block p-2 bg-blue-500 text-white rounded">Dashboard</Link></li>
          <li><Link to="/deepfake-detection" className="block p-2 hover:bg-gray-300 rounded">Deepfake Detection</Link></li>
          <li><Link to="/result-output" className="block p-2 hover:bg-gray-300 rounded">Result Output</Link></li>
        </ul>
      </nav>
    </div>
  );
}
