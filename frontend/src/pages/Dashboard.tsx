import React from 'react';
import FileUpload from '../components/FileUpload';

const Dashboard: React.FC = () => {
  return (
    <div className="p-6">
      <h1 className="text-3xl mb-6">Legal Intelligence Dashboard</h1>
      <FileUpload />
    </div>
  );
};

export default Dashboard;