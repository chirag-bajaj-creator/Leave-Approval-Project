import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import random

from datetime import datetime
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from google.generativeai import GenerativeModel, configure
from google.api_core.exceptions import GoogleAPIError, ResourceExhausted

# 2. Preprocessing: Load & Clean Data
def preprocess_data(csv_file):
    print("Loading and preprocessing data...")
    # Data Loading
    sandwich_data = pd.read_csv(csv_file)
    
    # Data Print
    print(sandwich_data.head(10))
    
    # Null Value Check 
    print(sandwich_data.isnull().sum())
    
    # Dtype Checking 
    print(sandwich_data.dtypes)
    
    # Duplicate Check
    duplicate_rows = sandwich_data[sandwich_data.duplicated()]
    num_duplicates = duplicate_rows.shape[0]
    print(f"Number of duplicate rows: {num_duplicates}")
    if num_duplicates > 0:
        print(duplicate_rows)
    
    # Checking Unique Values
    print(sandwich_data.nunique())
    
    
    # Outlier Handling
    def handle_outliers_winsorization(data, columns, threshold=3):
        # Apply Winsorization (Capping extreme values) directly on the dataset
        for col in columns:
            mean = data[col].mean()
            std = data[col].std()

            # Define upper and lower limits based on Z-score threshold
            lower_limit = mean - (threshold * std)
            upper_limit = mean + (threshold * std)

            # Cap values at lower and upper limits directly in the original dataset
            data[col] = np.where(data[col] < lower_limit, lower_limit, data[col])
            data[col] = np.where(data[col] > upper_limit, upper_limit, data[col])

    # Define columns to handle outliers
    selected_columns = ["Tasks Completed on Time", "Missed Deadlines", "Leaves in Team for Same Period", "Eligibility Score"]

    # Apply Winsorization to handle outliers directly in the dataset
    handle_outliers_winsorization(sandwich_data, selected_columns, threshold=3)

    # Print the first few rows of the cleaned dataset
    print("Cleaned Sandwich Data (Outliers Handled):")
    print(sandwich_data.head())
    
    
    # Label Encoding
    columns_to_encode = [
        'Department/Team', 'Designation/Role', 'Current Employment Status', 
        'Leave Type', 'Weekend Leaves (Sat/Sun Included?)', 
        'Project Involvement', 'Leave Impact on Work', 'Leave Approval Status'
    ]

    # Dictionary to store label encodings
    label_mappings = {}

    # Apply label encoding only to selected columns
    for col in columns_to_encode:
        if col in sandwich_data.columns:  # Ensure column exists in the dataset
            le = LabelEncoder()
            sandwich_data[col] = le.fit_transform(sandwich_data[col])
            label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    # Print the transformed dataset
    print("Label Encoded Employee Data (First 5 Rows):")
    print(sandwich_data.head())

    # Print the label encoding mappings for reference
    print("\nLabel Encoding Mappings:")
    for col, mapping in label_mappings.items():
        print(f"{col}: {mapping}")
        
    # Feature Engineering
    # Ensure "Date of Joining" is in datetime format
    sandwich_data["Date of Joining"] = pd.to_datetime(sandwich_data["Date of Joining"], errors='coerce')

    # Feature interaction: Multiply Skill Rating by Past Performance Reviews
    sandwich_data["Skill_Performance"] = sandwich_data["Skill Rating"] * sandwich_data["Past Performance Reviews"]

    # Extract month from the "Date of Joining"
    sandwich_data["Leave Month"] = sandwich_data["Date of Joining"].dt.month

    # Avoid Division Errors by adding 1 to the denominator
    sandwich_data["Leave Approval Rate"] = (sandwich_data["Leave Approval Status"] / (sandwich_data["Leave Frequency"] + 1)) * 100

    # Compute Leave Impact Score (Higher values mean more impact)
    sandwich_data["Leave Impact Score"] = (sandwich_data["Missed Deadlines"] + (1 - sandwich_data["Tasks Completed on Time"])) * sandwich_data["Leave Frequency"]

    # Compute Team Workload Ratio (Fraction of active employees in the team)
    sandwich_data["Team Workload Ratio"] = sandwich_data["Active Employees in Team"] / sandwich_data["Team Size"]
    
    # Date of Joining Drop
    sandwich_data.drop(columns=["Date of Joining"], inplace=True)
    
    # Outlier Handling For New features
    selected_columns = [
        "Tasks Completed on Time", 
        "Missed Deadlines", 
        "Leaves in Team for Same Period", 
        "Leave Impact Score",
        "Skill_Performance",
    ]

    # Apply Winsorization to handle outliers directly in the dataset
    handle_outliers_winsorization(sandwich_data, selected_columns, threshold=3)

    # Print the first few rows of the cleaned dataset
    print("✅ Cleaned Sandwich Data (Outliers Handled):")
    print(sandwich_data.head())
    
    
    # VIF Factor
    X_numeric = sandwich_data.select_dtypes(include=[np.number]).dropna()
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_numeric.columns
    vif_data["VIF"] = [variance_inflation_factor(X_numeric.values, i) for i in range(X_numeric.shape[1])]
    print(vif_data.sort_values(by="VIF", ascending=False))
    
    # Check min, max, and quartiles to confirm Winsorization effect
    print(sandwich_data[selected_columns].describe())
    
    # Checking Imbalanced Dataset Or not
    target_column = "Leave Approval Status" if "Leave Approval Status" in sandwich_data.columns else "target"
    target_counts = sandwich_data[target_column].value_counts(normalize=True) * 100
    is_imbalanced = any(target_counts < 20)

    print("\n🔍 **Class Distribution (%):**")
    print(target_counts)

    if is_imbalanced:
        print("\n⚠️ The dataset is **IMBALANCED**! Consider handling imbalance using SMOTE or other techniques.")
    else:
        print("The dataset is **BALANCED**. No special imbalance handling is required.")

    # Check for columns to drop
    columns_to_check = ['Unnamed: 0', 'Eligibility Score', 'Sandwich Policy Applied?']
    for col in columns_to_check:
        if col in sandwich_data.columns:
            print(f"Column '{col}' exists in the DataFrame.")
        else:
            print(f"Column '{col}' does NOT exist in the DataFrame.")
            
    # Drop columns
    columns_to_drop = [col for col in ['Eligibility Score', 'Sandwich Policy Applied?'] if col in sandwich_data.columns]
    if columns_to_drop:
        sandwich_data = sandwich_data.drop(columns=columns_to_drop)
        
    # Save updated dataset
    sandwich_data.to_csv('sandwich_leave.csv', index=False)
    
    return sandwich_data

# 3. Sandwich Policy Detection
def detect_sandwich_leave(df):
    print("Detecting sandwich leave cases...")
    sandwich_data = df.copy()
    
    # Define Holidays for 2024
    holidays_2024 = [
        "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27",
        "2024-06-19", "2024-07-04", "2024-09-02", "2024-10-14",
        "2024-11-11", "2024-11-28", "2024-12-25"
    ]

    # Function to check if a date is a weekend
    def is_weekend(date_str):
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        return date_obj.weekday() >= 5  # 5 = Saturday, 6 = Sunday

    # Ensure "Leave Date" column is in datetime format
    sandwich_data["Leave Date"] = pd.to_datetime(sandwich_data["Leave Date"], errors='coerce')

    # Get the complete list of employees (including duplicates)
    employee_ids = sandwich_data["Employee ID"].tolist()

    # Function for Sandwich Leave Detection
    def check_sandwich_leave():
        sandwich_leaves = []
        leave_dates_by_employee = sandwich_data.groupby("Employee ID")["Leave Date"].apply(set).to_dict()
        
        for employee_id in employee_ids:
            emp_leaves = leave_dates_by_employee.get(employee_id, [])
            
            for leave_date in emp_leaves:
                date_str = leave_date.strftime("%Y-%m-%d")

                # Identify adjacent days
                prev_day = (leave_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                next_day = (leave_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                weekday = leave_date.weekday()

                is_sandwich = False
                sandwich_pattern = ""
                sandwich_dates = []

                # Rules for Sandwich Leave Detection
                if weekday == 0 and next_day in holidays_2024:
                    is_sandwich = True
                    sandwich_pattern = "Monday before Tuesday holiday"
                    sandwich_dates = [date_str, next_day]
                elif weekday == 4 and prev_day in holidays_2024:
                    is_sandwich = True
                    sandwich_pattern = "Friday after Thursday holiday"
                    sandwich_dates = [prev_day, date_str]
                elif weekday == 3 and next_day in holidays_2024:
                    is_sandwich = True
                    sandwich_pattern = "Thursday before Friday holiday"
                    sandwich_dates = [date_str, next_day]
                elif weekday == 1 and prev_day in holidays_2024:
                    is_sandwich = True
                    sandwich_pattern = "Tuesday after Monday holiday"
                    sandwich_dates = [prev_day, date_str]
                elif weekday == 2 and prev_day in holidays_2024 and next_day in holidays_2024:
                    is_sandwich = True
                    sandwich_pattern = "Bridge day between holidays"
                    sandwich_dates = [prev_day, date_str, next_day]
                elif weekday == 0 and is_weekend(prev_day):
                    is_sandwich = True
                    sandwich_pattern = "Monday extending weekend"
                    sandwich_dates = [prev_day, date_str]
                elif weekday == 4 and is_weekend(next_day):
                    is_sandwich = True
                    sandwich_pattern = "Friday extending weekend"
                    sandwich_dates = [date_str, next_day]

                if is_sandwich:
                    sandwich_leaves.append([employee_id, date_str, "sandwich leave", sandwich_pattern, sandwich_dates])
                else:
                    sandwich_leaves.append([employee_id, date_str, "non sandwich leave", "0", [date_str]])

        return sandwich_leaves

    # Run Sandwich Leave Detection
    sandwich_leaves = check_sandwich_leave()

    # Store Results in a DataFrame
    results_df = pd.DataFrame(sandwich_leaves, columns=["Employee ID", "Leave Date", "Leave Type", "Sandwich Pattern", "Connected Dates"])

    # Save Results to CSV
    results_df.to_csv("sandwich_leave_results.csv", index=False)

    print("✅ Sandwich Leave Detection Completed and saved to 'sandwich_leave_results.csv'")
    return results_df

# 4. Compliance Check
def compliance_check(df):
    print("Checking compliance against leave policies...")
    # Set Google API Key
    os.environ["GOOGLE_API_KEY"] = "Your google api key"
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing Google API key. Please set the environment variable 'GOOGLE_API_KEY'.")

    configure(api_key=api_key)

    # Load the dataset
    # Replace with actual file path

    # Sort dataset by Employee ID in ascending order
    df = df.sort_values(by="Employee ID")

    # Initialize Gemini AI Model
    model = GenerativeModel("gemini-1.5-pro-latest")

    def compliance_check_with_ai(employee_data):
        if employee_data.empty:
            return "No data found for the given Employee ID"
        
        employee_info = employee_data.to_dict(orient='records')

        system_message = """
        You are an HR compliance assistant. Your job is to analyze employee leave data
        and verify compliance based on the following:
        - Valid leave types
        - Leave approval status
        - Leave duration limits
        - Maternity leave eligibility (min 80 days of tenure)
        - Data privacy (ensure no sensitive data like Aadhaar/bank details is exposed)
        
        Provide results in a structured way with a compliance percentage.
        """
        
        user_prompt = f"""
        {system_message}
        Analyze the following employee leave data and check compliance:
        {employee_info}
        """
        
        max_retries = 6
        base_retry_delay = 5  # Initial retry delay in seconds
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(user_prompt)
                return response.text
            except ResourceExhausted as e:
                if attempt < max_retries - 1:
                    wait_time = base_retry_delay * (2 ** attempt) + random.uniform(0, 5)
                    print(f"Quota exceeded. Retrying in {int(wait_time)} seconds...")
                    time.sleep(wait_time)
                else:
                    return "Quota exceeded. Please check your Google Cloud quota limits or try again later."
            except GoogleAPIError as e:
                return f"An error occurred: {str(e)}"

    # Main function to check compliance for the first 10 employees
    def main():
        first_10_employees = df["Employee ID"].unique()[:10]
        results = []  # List to store results

        for index, employee_id in enumerate(first_10_employees):
            print(f"Checking compliance for Employee ID: {employee_id}")
            employee_data = df[df['Employee ID'] == employee_id]
            compliance_results = compliance_check_with_ai(employee_data)
            print(f"Compliance Check Result for Employee {employee_id}:\n{compliance_results}\n")
            print("-" * 80)  # Separator for readability
            
            # Save result in list
            results.append({"Employee ID": employee_id, "Compliance Result": compliance_results})
            
            # **Pause after every 4 requests to avoid hitting the quota limit**
            if (index + 1) % 4 == 0 and index + 1 < len(first_10_employees):
                print("Pausing for 60 seconds to prevent API quota exhaustion...")
                time.sleep(60)  # Wait for 1 minute
        
        # Convert results to a DataFrame and save to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv("compliance_results_final.csv", index=False)
        print("Compliance results saved to compliance_results_final.csv")

    if __name__ == "__main__":
        main()
# 5. Score Calculation
def calculate_score(df):
    print("Calculating leave eligibility scores...")

    sandwich_data = df.copy()

    # Drop unnecessary columns
    drop_cols = ["Employee Name", "Leave Date"]
    sandwich_data_cleaned = sandwich_data.drop(columns=[col for col in drop_cols if col in sandwich_data.columns], errors='ignore')

    # Select relevant features
    feature_cols = [col for col in [
        "Employee Performance Score", "Skill Rating", "Past Performance Reviews",
        "Performance Rating", "Total Work Duration (years)", "Project Involvement",
        "Leave Impact on Work", "Leave Frequency", "Total Leave Days (per month/year)"
    ] if col in sandwich_data_cleaned.columns]

    # Ensure there are numerical columns left to work with
    if len(feature_cols) == 0:
        raise ValueError("No valid numeric features found for score calculation.")

    sandwich_data_filtered = sandwich_data_cleaned[feature_cols]

    # Define weights for manual scoring (ensure they sum up to ~1 for balance)
    weights = {
        "Employee Performance Score": 0.3,
        "Skill Rating": 0.2,
        "Performance Rating": 0.15,
        "Past Performance Reviews": 0.1,
        "Leave Impact on Work": -0.05,  # Penalizing leaves impacting work
        "Leave Frequency": -0.1,  # Penalizing frequent leaves
        "Total Leave Days (per month/year)": -0.05,
        "Total Work Duration (years)": 0.05  # Reduced weight, since tenure shouldn't be overpowered
    }

    # Compute weighted sum for available columns
    y = np.zeros(len(sandwich_data_filtered))
    for col in feature_cols:
        if col in weights:
            y += sandwich_data_filtered[col].fillna(sandwich_data_filtered[col].mean()) * weights[col]

    # Normalize `y` using min-max scaling (avoid complete compression)
    y_min, y_max = y.min(), y.max()
    if y_max > y_min:
        y = (y - y_min) / (y_max - y_min)
    else:
        y = np.zeros_like(y)  # Avoid division by zero

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(sandwich_data_filtered, y, test_size=0.2, random_state=42)

    # Define a neural network model with appropriate activation functions
    model = keras.Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.2),  # Dropout to prevent overfitting
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="linear")  # **Changed activation from ReLU to Linear**
    ])

    # Compile the model with mean squared error loss (since it's regression)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    # Train the model with early stopping
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Function to predict eligibility score
    def predict_score(input_data):
        input_array = np.array(input_data).reshape(1, -1)
        predicted = model.predict(input_array)[0][0] * 100  # Scale back to 0-100
        return int(round(max(0, min(100, predicted))))  # Convert to int and ensure valid range

    # Batch processing function for large datasets
    def process_in_batches(data, batch_size=500):
        predictions = []
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i+batch_size]
            batch_pred = model.predict(batch) * 100  # Scale back to 0-100
            batch_pred = np.clip(batch_pred.flatten(), 0, 100)  # Ensure scores remain within 0-100
            predictions.extend(batch_pred.astype(int))
        return predictions

    # Apply batch processing to dataset
    sandwich_data["Predicted Eligibility Score"] = process_in_batches(sandwich_data_filtered)

    # Save updated dataset
    sandwich_data.to_csv("updated_emp_sandwich.csv", index=False)
    print("✅ Updated dataset saved as updated_emp_sandwich.csv")

    return sandwich_data, model


# 6. Model Save (if ML model is used for scoring)
def save_model(model, filename="leave_model.pkl"):
    print("Saving trained model...")
    # Save the trained model as a .h5 file
    model.save("employee_eligibility_model.h5")
    print("Model saved as employee_eligibility_model.h5")
    return True

# 7. Merge Data (if required)
def merge_data(df1, df2):
    print("Merging datasets...")
    # Ensure df1 and df2 are DataFrames
    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        print("Error: Input data must be pandas DataFrames")
        return None
    
    # Exclude the 'Leave Date' column from df2
    columns_to_exclude = ["Leave Date"]
    df2_filtered = df2.drop(columns=[col for col in columns_to_exclude if col in df2.columns], errors="ignore")

    # Merge the datasets on Employee ID
    merged_df = pd.merge(df1, df2_filtered, on="Employee ID", how="left")

    # Save the merged dataset
    merged_file_path = "merged_sandwich_leave_data.csv"
    merged_df.to_csv(merged_file_path, index=False)

    print(f"✅ Merged dataset saved as {merged_file_path}")

    return merged_df

# 8. Pre-Approval Agent (Send for approval based on score)
def pre_approval_agent(df):
    print("Running pre-approval AI agent...")
    # Replace with your OpenAI API key
    client = OpenAI(api_key="Your open api key")

    # Define holidays for correct sandwich leave detection
    HOLIDAYS_2024 = [
        "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27",
        "2024-06-19", "2024-07-04", "2024-09-02", "2024-10-14",
        "2024-11-11", "2024-11-28", "2024-12-25"
    ]

    def get_employee_data(csv_path):
        if not os.path.exists(csv_path):
            print(f"Error: File '{csv_path}' not found!")
            return {}
        
        df = pd.read_csv(csv_path)
        df = df.sort_values(by=["Employee ID"]).reset_index(drop=True)  # Ensuring fixed order
        grouped_data = df.groupby("Employee ID").agg(list).reset_index()
        return grouped_data.to_dict(orient="records")[:10]  # Process first 20 employees consistently

    def is_weekend(date_str):
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        return date_obj.weekday() in [5, 6]  # Saturday or Sunday

    def identify_sandwich_sets(leave_dates):
        leave_dates = sorted(set([date for sublist in leave_dates for date in eval(sublist)]),
                            key=lambda x: datetime.strptime(x, "%Y-%m-%d"))
        
        sandwich_sets = []
        temp_set = []
        for i, date in enumerate(leave_dates):
            date_obj = datetime.strptime(date, "%Y-%m-%d").date()
            prev_day = (date_obj - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            next_day = (date_obj + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            
            # If previous or next day is a holiday or weekend, it's a sandwich leave
            if prev_day in HOLIDAYS_2024 or next_day in HOLIDAYS_2024 or is_weekend(prev_day) or is_weekend(next_day):
                temp_set.append(date)
            else:
                if len(temp_set) > 1:
                    sandwich_sets.append(temp_set)
                temp_set = [date]
        
        if len(temp_set) > 1:
            sandwich_sets.append(temp_set)
        
        return sandwich_sets if sandwich_sets else None  # Return None if no sandwich leave sets found

    def determine_leave_type(leave_frequency, performance_score):
        """
        Determines whether the leave is a full day or half day based on employee metrics.
        """
        if leave_frequency <= 2 and performance_score >= 75:
            return "Half Day"
        return "Full Day"

    def improve_explanation(decision, leave_dates, sandwich_sets):
        """
        Generates a structured explanation for leave approval or rejection.
        """
        sandwich_status = "Sandwich Leave" if sandwich_sets else "Non-Sandwich Leave"
        detected_sandwich_leaves = sandwich_sets if sandwich_sets else "No sandwich leave detected"
        
        if "Approved" in decision:
            return (
                f"Based on the provided information, the leave request has been *approved*.\n"
                f"Leave Dates: {leave_dates}\n"
                f"Detected Sandwich Leave Sets: {detected_sandwich_leaves}\n"
                f"Leave Category: {sandwich_status}\n"
                "The employee met the required performance and leave frequency criteria."
            )
        else:
            return (
                f"Based on the provided information, the leave request has been *rejected*.\n"
                f"Leave Dates: {leave_dates}\n"
                f"Detected Sandwich Leave Sets: {detected_sandwich_leaves}\n"
                f"Leave Category: {sandwich_status}\n"
                "The employee's leave frequency, performance score, or detected sandwich leave did not meet the approval criteria."
            )

    def genai_pre_approval_agent(csv_path):
        employee_data = get_employee_data(csv_path)
        if not employee_data:
            return {}
        
        pre_approved_requests = []
        
        for data in employee_data:
            emp = data["Employee ID"]
            leave_dates = data["Connected Dates"]
            sandwich_sets = identify_sandwich_sets(leave_dates)
            sandwich_status = "Sandwich" if sandwich_sets else "Non-Sandwich"

            leave_type = determine_leave_type(float(data['Leave Frequency'][0]), float(data['Employee Performance Score'][0]))
            ai_decision = "Rejected" if sandwich_sets and float(data['Employee Performance Score'][0]) < 60 else "Approved"
            
            improved_explanation = improve_explanation(ai_decision, ", ".join(leave_dates), sandwich_sets)

            pre_approved_requests.append({
                "Employee ID": emp,
                "Leave Dates": ", ".join(leave_dates),
                "AI Decision": ai_decision,
                "Explanation": improved_explanation,
                "Leave Type": leave_type if ai_decision == "Approved" else "N/A",
                "Detected Sandwich Leave Sets": sandwich_sets if sandwich_sets else "None",
                "Sandwich Status": sandwich_status
            })

        return pre_approved_requests

    def save_genai_decisions(pre_approved_requests):
        if not pre_approved_requests:
            print("No Generative AI pre-approved requests to save.")
            return

        decisions_df = pd.DataFrame(pre_approved_requests)
        decisions_df.to_csv("genai_preapproval_decision.csv", index=False)
        print("Generative AI Final Decisions Saved to genai_preapproval_decision.csv")

    if __name__ == "__main__":
        csv_path = "merged_sandwich_leave_data.csv"
        pre_approved_requests = genai_pre_approval_agent(csv_path)
        save_genai_decisions(pre_approved_requests)

# 9. Calculate Average Score
def calculate_avg_score(df):
    print("Calculating average leave scores for employees...")
    
    # Check if the DataFrame has the required column
    column_name = "Predicted Eligibility Score"
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in the dataset")
        return None
    
    # Compute the average value of the column
    avg_score = df[column_name].mean()
    
    print(f"Average Predicted Eligibility Score: {avg_score}")
    
    # Create a DataFrame with the average score
    avg_score_df = pd.DataFrame({'Average Eligibility Score': [avg_score]})
    avg_score_df.to_csv("average_scores.csv", index=False)
    
    return avg_score_df

# Pipeline Execution Flow
def pipeline(csv_file):

    
    # Load and preprocess data
    df = preprocess_data(csv_file)  # Assuming employee.csv is the input file
    
    # Processing steps
    sandwich_results = detect_sandwich_leave(df)
    compliance_results = compliance_check(df)
    df, model = calculate_score(df)
    
    # Save model
    save_model(model)
    
    # Merge data
    merged_df = merge_data(df, sandwich_results)
    
    # Pre-Approval Processing
    approval_results = pre_approval_agent(merged_df)
    
    # Calculate average score
    avg_scores = calculate_avg_score(df)
    
    print("Pipeline execution complete!")



import json

def get_employee_details(employee_id, file_path="genai_preapproval_decision.csv"):
    """
    Fetches employee details from gen_ai.csv based on Employee ID.
    Returns a tuple: (JSON response, status code).
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path, dtype=str)  # Read all columns as strings to avoid type mismatch

        # Ensure 'Employee_ID' column exists
        if 'Employee ID' not in df.columns:
            response = {"error": "Missing 'Employee_ID' column in CSV."}
            return json.dumps(response), 500

        # Standardize Employee_ID column
        df['Employee ID'] = df['Employee ID'].astype(str).str.strip()

        # Standardize input Employee ID
        employee_id = str(employee_id).strip()

        # Filter the data for the given employee ID
        employee_data = df[df['Employee ID'] == employee_id]

        # Check if the employee exists
        if employee_data.empty:
            response = {"error": f"No records found for Employee ID: {employee_id}"}
            return json.dumps(response), 404
        else:
            response = employee_data.to_dict(orient="records")  # Convert to JSON-compatible format
            return json.dumps(response, indent=4), 200  # Pretty JSON formatting

    except FileNotFoundError:
        response = {"error": f"The file {file_path} was not found."}
        return json.dumps(response), 500
    except Exception as e:
        response = {"error": str(e)}
        return json.dumps(response), 500





