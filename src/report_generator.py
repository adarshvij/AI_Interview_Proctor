"""
Report Generator Module

Analyzes infraction lists, aggregates violation counts, charts behavioral
anomalies using matplotlib, and writes final exam summary reports.
"""

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any

class ReportGenerator:
    """
    Assembles post-interview analytics logs. Generates summaries, visualizes
    distracted timelines, and exports report documentation for recruiters.
    """

    def __init__(self, candidate_name: str, candidate_id: str, logs: List[Dict[str, Any]], reports_dir: str = "reports"):
        """
        Initializes the ReportGenerator.

        Args:
            candidate_name (str): Full name of the test taker.
            candidate_id (str): Unique candidate identifier code.
            logs (List[Dict[str, Any]]): Collected logs from WarningSystem.
            reports_dir (str): Directory where generated reports are exported.
        """
        self.candidate_name = candidate_name
        self.candidate_id = candidate_id
        self.logs = logs
        self.reports_dir = reports_dir
        
        # Ensure target folder exists
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_report(self) -> str:
        """
        Aggregates proctoring results, generates statistical charts,
        and saves a text summary and CSV report file.

        Returns:
            str: Path to the generated report folder or summary file.
        """
        if not self.logs:
            # Create a mock log if empty to demonstrate functionality
            self.logs = [
                {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "type": "System Check", "description": "Candidate session started", "screenshot": ""}
            ]

        # Convert to pandas DataFrame for quick calculations
        df = pd.DataFrame(self.logs)
        
        # Export CSV log
        csv_filename = f"{self.candidate_id}_violation_report.csv"
        csv_filepath = os.path.join(self.reports_dir, csv_filename)
        df.to_csv(csv_filepath, index=False)
        
        # Generate visual charts
        self._generate_charts(df)
        
        # Generate summary text report
        txt_filename = f"{self.candidate_id}_summary.txt"
        txt_filepath = os.path.join(self.reports_dir, txt_filename)
        
        with open(txt_filepath, "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write("          AI PROCTORING ASSESSMENT REPORT         \n")
            f.write("==================================================\n\n")
            f.write(f"Candidate Name : {self.candidate_name}\n")
            f.write(f"Candidate ID   : {self.candidate_id}\n")
            f.write(f"Date           : {time.strftime('%Y-%m-%d')}\n\n")
            f.write("Violation Summary:\n")
            
            # Count events
            if "type" in df.columns:
                counts = df["type"].value_counts()
                for v_type, count in counts.items():
                    f.write(f" - {v_type}: {count} occurrences\n")
            else:
                f.write(" - No violations logged.\n")
                
            f.write("\nDetailed Log of Events:\n")
            for index, row in df.iterrows():
                f.write(f"[{row['timestamp']}] {row['type']}: {row['description']}\n")
                
            f.write("\n==================================================\n")
            f.write("                 End of Report                    \n")
            f.write("==================================================\n")

        return txt_filepath

    def _generate_charts(self, df: pd.DataFrame) -> None:
        """
        Saves visual analytical graphics mapping candidate anomalies to the reports directory.
        """
        if df.empty or "type" not in df.columns:
            return
            
        try:
            # Bar chart of violation categories
            plt.figure(figsize=(8, 4))
            df["type"].value_counts().plot(kind="bar", color="crimson")
            plt.title("Violation Frequencies")
            plt.xlabel("Violation Type")
            plt.ylabel("Incident Count")
            plt.tight_layout()
            
            chart_path = os.path.join(self.reports_dir, f"{self.candidate_id}_chart.png")
            plt.savefig(chart_path)
            plt.close()
        except Exception:
            # Fail silently during placeholder visualization execution
            pass
