import os
import json
import warnings
import numpy as np
import pandas as pd
import datetime
import ast
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

class SandwichLeavePipeline:
    """
    A comprehensive pipeline for sandwich leave detection, compliance checking, 
    and automated leave approval.
    """
    
    def __init__(self, config_path=None):
        """Initialize the pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.holidays = self.config.get("holidays", [
            "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27",
            "2024-06-19", "2024-07-04", "2024-09-02", "2024-10-14",
            "2024-11-11", "2024-11-28", "2024-12-25"
        ])
    
    def _load_config(self, config_path):
        """Load configuration from JSON file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            "input_file": "employee.csv",
            "output_dir": "output",
            "model_dir": "models",
            "batch_size": 32,
            "test_size": 0.2,
            "random_state": 42,
            "epochs": 100,
            "learning_rate": 0.001,
            "threshold_score": 60
        }
    
    def setup(self):
        """Set up the pipeline environment."""
        os.makedirs(self.config["output_dir"], exist_ok=True)
        os.makedirs(self.config["model_dir"], exist_ok=True)
        print(f"✅ Pipeline setup complete. Output directory: {self.config['output_dir']}")
    
    def load_data(self, file_path=None):
        """Load dataset from CSV file."""
        if file_path is None:
            file_path = self.config["input_file"]
        
        self.data = pd.read_csv(file_path)
        print(f"✅ Loaded data from {file_path} with {len(self.data)} records")
        return self.data
    
    def detect_sandwich_leaves(self):
        """Detect sandwich leaves in the dataset."""
        print("Starting sandwich leave detection...")
        
        if "Leave Date" not in self.data.columns or "Employee ID" not in self.data.columns:
            print("⚠️ Required columns missing for sandwich leave detection")
            return None
        
        sandwich_data = self.data.copy()
        sandwich_data["Leave Date"] = pd.to_datetime(sandwich_data["Leave Date"], errors='coerce')
        employee_ids = sandwich_data["Employee ID"].tolist()
        
        sandwich_leaves = []
        leave_dates_by_employee = sandwich_data.groupby("Employee ID")["Leave Date"].apply(set).to_dict()

        for employee_id in employee_ids:
            emp_leaves = leave_dates_by_employee.get(employee_id, [])

            for leave_date in emp_leaves:
                date_str = leave_date.strftime("%Y-%m-%d")
                prev_day = (leave_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                next_day = (leave_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                weekday = leave_date.weekday()

                def is_weekend(date_str):
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                    return date_obj.weekday() >= 5

                is_sandwich = False
                sandwich_pattern = ""
                sandwich_dates = []

                if weekday == 0 and next_day in self.holidays:
                    is_sandwich = True
                    sandwich_pattern = "Monday before Tuesday holiday"
                    sandwich_dates = [date_str, next_day]
                elif weekday == 4 and prev_day in self.holidays:
                    is_sandwich = True
                    sandwich_pattern = "Friday after Thursday holiday"
                    sandwich_dates = [prev_day, date_str]
                elif weekday == 3 and next_day in self.holidays:
                    is_sandwich = True
                    sandwich_pattern = "Thursday before Friday holiday"
                    sandwich_dates = [date_str, next_day]
                elif weekday == 1 and prev_day in self.holidays:
                    is_sandwich = True
                    sandwich_pattern = "Tuesday after Monday holiday"
                    sandwich_dates = [prev_day, date_str]
                elif weekday == 2 and prev_day in self.holidays and next_day in self.holidays:
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

        results_df = pd.DataFrame(sandwich_leaves, 
                              columns=["Employee ID", "Leave Date", "Leave Type", 
                                       "Sandwich Pattern", "Connected Dates"])
        
        output_path = os.path.join(self.config["output_dir"], "sandwich_leave_results.csv")
        results_df.to_csv(output_path, index=False)
        print(f"✅ Sandwich Leave Detection complete. Saved to {output_path}")
        
        self.sandwich_results = results_df
        return results_df

    def run_genai_approval(self):
        """Run the pre-approval agent using generative AI."""
        print("Running GenAI pre-approval process...")

        if not hasattr(self, 'sandwich_results'):
            print("⚠️ No sandwich leave detection results available")
            return None
        
        data = self.sandwich_results.copy()
        
        pre_approved_requests = []
        
        for idx, row in data.iterrows():
            emp_id = row["Employee ID"]
            leave_dates = row["Connected Dates"]
            
            leave_dates_list = []
            for sublist in leave_dates:
                if isinstance(sublist, str):
                    if sublist.startswith("[") and sublist.endswith("]"):  # Ensure it's a list string
                        try:
                            parsed_list = ast.literal_eval(sublist)
                            if isinstance(parsed_list, list):
                                leave_dates_list.extend(parsed_list)
                        except (ValueError, SyntaxError):
                            print(f"⚠️ Warning: Could not parse leave dates: {sublist}")
                    else:
                        # Directly append single date strings
                        leave_dates_list.append(sublist)
                elif isinstance(sublist, list):
                    leave_dates_list.extend(sublist)  # If already a list, extend it

            sandwich_status = "Sandwich Leave" if row["Leave Type"] == "sandwich leave" else "Non-Sandwich Leave"
            
            ai_decision = "Rejected" if sandwich_status == "Sandwich Leave" else "Approved"

            pre_approved_requests.append({
                "Employee ID": emp_id,
                "Leave Dates": str(leave_dates_list),
                "AI Decision": ai_decision,
                "Sandwich Status": sandwich_status
            })

        approval_df = pd.DataFrame(pre_approved_requests)
        output_path = os.path.join(self.config["output_dir"], "genai_pre_approval.csv")
        approval_df.to_csv(output_path, index=False)
        print(f"✅ GenAI pre-approval complete. Saved to {output_path}")
        
        self.approval_results = approval_df
        return approval_df


if __name__ == "__main__":
    pipeline = SandwichLeavePipeline("pipeline_config.json")
    pipeline.setup()
    pipeline.load_data()
    pipeline.detect_sandwich_leaves()
    pipeline.run_genai_approval()
