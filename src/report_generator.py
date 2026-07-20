"""
Report Generator Module

Generates a final summary CSV report of the interview session using pandas.
Provides reusable methods to structure the data and export it cleanly.
"""

import pandas as pd
import os
import time
from typing import Dict, Any

class ReportGenerator:
    """
    Handles the aggregation of session metrics into structured pandas DataFrames
    and exports them to CSV for HR/Proctor review.
    """

    def __init__(self, reports_dir: str = "reports"):
        """
        Initializes the ReportGenerator.

        Args:
            reports_dir (str): Directory where CSV reports will be saved.
        """
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def compile_data(self, 
                     duration_seconds: float, 
                     counters: Dict[str, int], 
                     risk_level: str) -> pd.DataFrame:
        """
        Structures the raw session data into a pandas DataFrame.

        Args:
            duration_seconds (float): Total length of the interview in seconds.
            counters (Dict[str, int]): The violation counters from WarningSystem.
            risk_level (str): The final categorical risk classification.

        Returns:
            pd.DataFrame: A single-row DataFrame containing the formatted summary.
        """
        # Format the duration nicely (e.g., MM:SS)
        mins, secs = divmod(int(duration_seconds), 60)
        formatted_duration = f"{mins:02d}:{secs:02d}"

        # Construct the data dictionary exactly matching our requirements
        data = {
            "Date": [time.strftime("%Y-%m-%d %H:%M:%S")],
            "Duration": [formatted_duration],
            "Phone_Detections": [counters.get("phone_usage", 0)],
            "Gaze_Violations": [counters.get("looking_away", 0)],
            "Multiple_Faces": [counters.get("multiple_faces", 0)],
            "No_Face_Violations": [counters.get("no_face", 0)],
            "Final_Risk_Level": [risk_level]
        }

        # Create and return the pandas DataFrame
        df = pd.DataFrame(data)
        return df

    def export_to_csv(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Saves the structured DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): The DataFrame to export.
            filename (str, optional): Custom filename. Defaults to a timestamped name.

        Returns:
            str: The filepath of the saved CSV.
        """
        if filename is None:
            filename = f"interview_report_{int(time.time())}.csv"
            
        filepath = os.path.join(self.reports_dir, filename)
        
        # Save to CSV (index=False prevents pandas from writing the row numbers)
        df.to_csv(filepath, index=False)
        
        return filepath
