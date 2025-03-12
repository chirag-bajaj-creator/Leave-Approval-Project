import pandas as pd
import datetime

# Load Holidays
HOLIDAYS_2024 = [
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-10-14",
    "2024-11-11", "2024-11-28", "2024-12-25"
]

def is_weekend(date_str):
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    return date_obj.weekday() in [5, 6]  # Saturday or Sunday

def check_sandwich_leave(file_path):
    sandwich_data = pd.read_csv(file_path)
    sandwich_data["Leave Date"] = pd.to_datetime(sandwich_data["Leave Date"], errors='coerce')
    
    employee_ids = sandwich_data["Employee ID"].unique()
    sandwich_leaves = []
    leave_dates_by_employee = sandwich_data.groupby("Employee ID")["Leave Date"].apply(set).to_dict()
    
    for employee_id in employee_ids:
        emp_leaves = leave_dates_by_employee.get(employee_id, [])
        
        for leave_date in emp_leaves:
            date_str = leave_date.strftime("%Y-%m-%d")
            prev_day = (leave_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            next_day = (leave_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            
            is_sandwich = False
            sandwich_pattern = ""
            sandwich_dates = []
            
            if prev_day in HOLIDAYS_2024 and next_day in HOLIDAYS_2024:
                is_sandwich = True
                sandwich_pattern = "Bridge day between holidays"
                sandwich_dates = [prev_day, date_str, next_day]
            elif prev_day in HOLIDAYS_2024 or next_day in HOLIDAYS_2024:
                is_sandwich = True
                sandwich_pattern = "Leave adjacent to holiday"
                sandwich_dates = [prev_day, date_str] if prev_day in HOLIDAYS_2024 else [date_str, next_day]
            elif is_weekend(prev_day) or is_weekend(next_day):
                is_sandwich = True
                sandwich_pattern = "Weekend extended leave"
                sandwich_dates = [prev_day, date_str] if is_weekend(prev_day) else [date_str, next_day]
            
            leave_type = "sandwich leave" if is_sandwich else "non sandwich leave"
            sandwich_leaves.append([employee_id, date_str, leave_type, sandwich_pattern, sandwich_dates])
    
    results_df = pd.DataFrame(sandwich_leaves, columns=["Employee ID", "Leave Date", "Leave Type", "Sandwich Pattern", "Connected Dates"])
    results_df.to_csv("data/sandwich_leave_results.csv", index=False)
    print("✅ Sandwich Leave Detection Completed and saved to 'data/sandwich_leave_results.csv'")

if __name__ == "__main__":
    check_sandwich_leave("data/processed_sandwich_leave.csv")
