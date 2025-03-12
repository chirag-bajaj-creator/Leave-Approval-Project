from flask import Flask, request, jsonify
import tensorflow as tf
import os
import pandas as pd
import json
from werkzeug.utils import secure_filename
from fixed_leave_pipeline import pipeline, get_employee_details
from flasgger import Swagger
from flask_socketio import SocketIO, emit
from twilio.rest import Client
from flask_cors import CORS
import numpy as np
from twilio.twiml.messaging_response import MessagingResponse
import time

app = Flask(__name__)
CORS(app)
app.config['SWAGGER'] = {
    'title': 'Leave Prediction API',
    'uiversion': 3,
    'specs_route': '/apidocs/'
}
swagger = Swagger(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Twilio Credentials
TWILIO_ACCOUNT_SID = "Your Twillo SID"
TWILIO_AUTH_TOKEN = "Twilio Auth Key"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14*********"
HR_WHATSAPP_NUMBER = "whatsapp:+919876543210"
EMPLOYEE_WHATSAPP_NUMBER = "whatsapp:+911234567890"

# Initialize Twilio Client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Store pending leave requests and employee phone numbers
leave_requests = {}
employee_phone_numbers = {}

@app.route('/employee-details/<employee_id>', methods=['GET'])
def employee_details(employee_id):
    """
    Get employee details by Employee ID
    ---
    tags:
      - Employee
    parameters:
      - name: employee_id
        in: path
        type: string
        required: true
        description: Employee ID to fetch details
    responses:
      200:
        description: Employee details found
        schema:
          type: object
      404:
        description: Employee not found
      500:
        description: Internal Server Error
    """
    try:
        response_json, status_code = get_employee_details(employee_id)
        print("DEBUG: API Response =", response_json)

        # Check if employee data exists
        if not response_json:
            return jsonify({"status": "error", "message": "Employee details not found"}), 404

        # Ensure response is valid JSON
        if isinstance(response_json, str):
            try:
                response_json = json.loads(response_json)
            except json.JSONDecodeError:
                return jsonify({"status": "error", "message": "Invalid JSON format in response"}), 500

        # Ensure response is a dictionary or list of dictionaries
        if isinstance(response_json, dict):
            response_json = [response_json]
        elif not isinstance(response_json, list) or not all(isinstance(item, dict) for item in response_json):
            return jsonify({"status": "error", "message": "Invalid response format from backend"}), 500

        # Convert list to DataFrame
        df = pd.DataFrame(response_json)

        # ✅ Replace NaN values with None (valid JSON format)
        df = df.replace({np.nan: None})

        # Extract first employee record
        employee_details = df.iloc[0].to_dict()

        # Store leave request and employee phone number
        leave_requests[employee_details['Employee ID']] = employee_details
        employee_phone_numbers[employee_details['Employee ID']] = EMPLOYEE_WHATSAPP_NUMBER

        # Send leave request to HR via WhatsApp
        send_leave_request_to_hr(employee_details['Employee ID'], employee_details)

        return jsonify({"status": "success", "data": employee_details}), status_code

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500
        


# AI Sends Leave Request to HR
def send_leave_request_to_hr(employee_id, leave_details):
    message_body = f"""
    *Leave Request for Approval* 👤
    Employee ID: {employee_id}
    Leave Dates: {leave_details['Leave Dates']}
    AI Decision: {leave_details['AI Decision']}
    Explanation: {leave_details['Explanation']}
    Leave Type: {leave_details['Leave Type']}
    Sandwich Status: {leave_details['Sandwich Status']}
    
    Reply with:
    ✅ *ACCEPT {employee_id}* to approve
    ❌ *REJECT {employee_id}* to decline
    """
    
    message = client.messages.create(
        from_=TWILIO_WHATSAPP_NUMBER,
        body=message_body,
        to=HR_WHATSAPP_NUMBER
    )
    print(f"✅ Leave request sent to HR. Message SID: {message.sid}")

@app.route("/twilio-webhook", methods=['POST'])
def twilio_webhook():
    """
    Handle manager's approval/rejection via WhatsApp.
    ---
    tags:
      - Twilio
    responses:
      200:
        description: Manager response received successfully
      400:
        description: Invalid format
    """
    incoming_msg = request.form.get('Body', '').strip().upper()
    words = incoming_msg.split()
    response = MessagingResponse()
    
    if len(words) == 2 and words[0] in ["ACCEPT", "REJECT"]:
        action, employee_id = words
        
        if employee_id in leave_requests:
            decision = "approved" if action == "ACCEPT" else "rejected"
            notify_employee(employee_id, decision)
            del leave_requests[employee_id]
            response.message(f"✅ Leave request for Employee ID {employee_id} has been {decision}.")
        else:
            response.message("❌ Invalid Employee ID. No pending leave request found.")
    else:
        response.message("⚠ Invalid response format. Use ACCEPT <EMP_ID> or REJECT <EMP_ID>.")

    return str(response)

# Notify Employee of HR's Decision
def notify_employee(employee_id, decision):
    message_body = f"📢 *Leave Status Update*\nYour leave request has been *{decision}* by HR."
    message = client.messages.create(
        from_=TWILIO_WHATSAPP_NUMBER,
        body=message_body,
        to=EMPLOYEE_WHATSAPP_NUMBER
    )
    print(f"✅ Employee notified. Message SID: {message.sid}")


app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')

# Ensure the uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/model-training', methods=['POST'])
def train_model():
    """
    Upload a CSV file and trigger model training with real-time progress updates
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    socketio.emit('progress', {"status": "📤 Upload successful, starting model training...", "progress": 10})

    try:
        # ✅ Simulating training progress using a loop
        for i in range(20, 101, 20):
            time.sleep(1)  # Simulate training process
            socketio.emit('progress', {"status": f"Training... {i}%", "progress": i})

        # ✅ Run pipeline after simulated progress
        df = pipeline(file_path)

        # ✅ Ensure final event is emitted after training completes
        socketio.emit('progress', {
            "status": "✅ Model training complete!",
            "progress": 100,
            "data": df  # Convert DataFrame to JSON format
        })

        return jsonify({"message": "Model Trained Successfully", "download": df})

    except Exception as e:
        socketio.emit('progress', {"status": f"❌ Error during training: {str(e)}", "progress": 0})
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=3000, debug=True)