# Leave Approval System

## 🚀 Project Overview
The **Leave Approval System** is an automated platform designed to streamline the process of leave requests and approvals within an organization. The system provides role-based access, allowing employees to request leave and managers to approve or reject requests efficiently.

## 📌 Features
- **Leave Request Submission**: Employees can request leave by selecting dates, type of leave, and providing a reason.
- **Approval & Rejection Process**: Managers can review, approve, or reject leave requests.
- **Leave Balance Tracking**: Employees can check their available leave balance.
- **Notification System**: Users receive email/SMS notifications for status updates.
- **Dashboard & Reports**: Admins can monitor leave trends and generate reports.

## 🛠️ Tech Stack
- **Frontend**: React.js
- **Backend**: Flask
- **Database**: MySQL
- **Deployment**: Docker, Informatica: AI Powered Cloud Data Management

## 📂 Project Structure
```
leave-approval-system/
│── backend/            # Backend API (Flask, Twilio)
│── frontend/           # Frontend application (React.js)
│── database/           # Database scripts and configurations
│── README.md           # Project documentation
```

## 🔧 Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/Deepanshu-Sehgal/Leave-Approval-Project.git
cd leave-approval-system
```

### 2️⃣ Backend Setup
```sh
cd final_pipeline
pip install   # Install dependencies 
python backend_leave_pipeline.py    # Run the backend server
```

### 3️⃣ Frontend Setup
```sh
cd frontend
npm install   # Install dependencies
npm start     # Run the frontend app
```

### 4️⃣ Environment Configuration
- Update database credentials, and API keys.


## 📜 API Endpoints
| Method | Endpoint                                           | Description               |
|--------|----------------------------------------------------|---------------------------|
| POST   | http:localhost:3000'/employee-details/<employee_id | Apply for leave           |
| POST   | '/model-training'                                  | Real time model traning   |
| POST   | /twilio-webhook                                    | Approve leave/Reject Leave|


## 🚀 Future Enhancements
- Mobile app integration.
- More Fine Tuned AI-based leave prediction system.
- Automated HR analytics dashboard.

## 🏆 Hackathon Submission Details
- **Team Name**: [Your Team Name]
- **Hackathon Name**: [Hackathon Name]
- **Submission Date**: [Date]
- **Presentation**: [Link to PPT/Video]

## 📩 Contact
For any queries, feel free to reach out at **[deepanshu20@s.amity.edu]**.

---
💡 *This project is developed as part of a hackathon submission to showcase an efficient leave approval system.*
