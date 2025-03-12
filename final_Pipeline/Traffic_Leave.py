import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Securely fetch Google Maps API Key
GOOGLE_MAPS_API_KEY = "AIzaSyDS6WLGnfACzjZUrk2DQScMXzBB2CLcM0U"
GOOGLE_MAPS_API_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"

# In-memory storage for buffer time
employee_leave_status = {}

def check_traffic(employee_location, office_location):
    """
    Fetches real-time traffic data from Google Maps API.
    Returns travel delay in minutes or None if API fails.
    """
    params = {
        "origins": employee_location,
        "destinations": office_location,
        "key": GOOGLE_MAPS_API_KEY,
        "departure_time": "now",  # Required for real-time traffic
        "traffic_model": "best_guess",  # Options: best_guess, optimistic, pessimistic
        "mode": "driving"  # Ensures it's for vehicle travel (not walking or transit)
    }
    
    try:
        response = requests.get(GOOGLE_MAPS_API_URL, params=params)
        response.raise_for_status()  # Raises an error for HTTP failures

        data = response.json()

        # Debugging: Print full API response
        print("\n🔍 Google API Response:", data)

        if "rows" in data and data["rows"]:
            elements = data["rows"][0]["elements"]
            if elements and "duration_in_traffic" in elements[0]:
                duration = elements[0]["duration_in_traffic"]["value"] / 60  # Convert to minutes
                return duration
            else:
                print("⚠️ Traffic data unavailable in response.")
        else:
            print("⚠️ Invalid API response structure.")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching traffic data: {e}")

    return None  # Return None if API fails or no data found

def grant_buffer_time(employee_id, buffer_time):
    """
    Grants buffer time to an employee if traffic delay exceeds 20 minutes.
    """
    employee_leave_status[employee_id] = {
        "buffer_time": buffer_time,
        "leave_status": "Buffer Time Granted"
    }
    print(f"✅ Employee {employee_id}: Buffer time of {buffer_time} minutes granted.")

def approve_wfh(employee_id):
    """
    Approves Work From Home (WFH) if the traffic delay is extreme (>45 minutes).
    """
    employee_leave_status[employee_id] = {
        "buffer_time": 0,
        "leave_status": "WFH Approved"
    }
    print(f"✅ Employee {employee_id}: Work From Home (WFH) approved due to extreme traffic.")

def process_traffic_leave(employee_id, employee_location, office_location):
    """
    Processes the employee's leave request based on real-time traffic conditions.
    """
    traffic_time = check_traffic(employee_location, office_location)

    if traffic_time is None:
        print("❌ Failed to retrieve traffic data. Please try again later.")
        return
    
    normal_travel_time = 30  # Assume normal commute is 30 minutes
    delay = traffic_time - normal_travel_time

    print(f"\n🚦 Traffic Analysis for {employee_id}:")
    print(f"   🔹 Normal Travel Time: {normal_travel_time} mins")
    print(f"   🔹 Actual Travel Time: {traffic_time} mins")
    print(f"   🔹 Delay: {delay} mins\n")

    if delay < 20:
        print(f"✅ Employee {employee_id}: No leave required. Traffic is normal.")
    elif 20 <= delay < 45:
        grant_buffer_time(employee_id, delay)
    else:
        approve_wfh(employee_id)  # If delay is extreme, suggest WFH

# Example Execution
employee_id = "EMP123"
employee_location = "28.7041,77.1025"  # Latitude,Longitude format
office_location = "28.4595,77.0266"  # Latitude,Longitude format

process_traffic_leave(employee_id, employee_location, office_location)

# Check leave status
print("\n📌 Employee Leave Status:", employee_leave_status)
