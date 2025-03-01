import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QLineEdit, QGridLayout
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl

# -------------- Data Loading and Preprocessing --------------

# Ensure that the program is looking in the samme script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load the CSV files (assumed to be in the same directory)
data2023 = pd.read_csv("data2023.csv")
data2024 = pd.read_csv("data2024.csv")

# Ensure numeric conversion for grade counts and averages
grade_cols = ['7', '6', '5', '4', '3', '2', '1', 'P', 'N']
for col in grade_cols:
    data2023[col] = pd.to_numeric(data2023[col], errors='coerce').fillna(0)
    data2024[col] = pd.to_numeric(data2024[col], errors='coerce').fillna(0)

# Convert ENTRIES to numeric as well
data2023['ENTRIES'] = pd.to_numeric(data2023['ENTRIES'], errors='coerce').fillna(0)
data2024['ENTRIES'] = pd.to_numeric(data2024['ENTRIES'], errors='coerce').fillna(0)

# Convert averages to numeric (if missing, they remain NaN)
data2023['Average (School)'] = pd.to_numeric(data2023['Average (School)'], errors='coerce')
data2023['Average (Worldwide)'] = pd.to_numeric(data2023['Average (Worldwide)'], errors='coerce')
data2024['Average (School)'] = pd.to_numeric(data2024['Average (School)'], errors='coerce')
data2024['Average (Worldwide)'] = pd.to_numeric(data2024['Average (Worldwide)'], errors='coerce')

# For average comparisons, exclude subjects with missing average (e.g. DT in 2023)
data2023_avg = data2023.dropna(subset=['Average (School)', 'Average (Worldwide)'])
data2024_avg = data2024.dropna(subset=['Average (School)', 'Average (Worldwide)'])

# Get a sorted union of subjects (for dropdowns)
subjects_all = sorted(list(set(data2023["SUBJECT"].unique()).union(set(data2024["SUBJECT"].unique()))))

# -------------- visualisation Functions --------------

def plot_grade_distribution(subject):
    """
    For a given subject, plot a grouped bar chart comparing the grade distribution (in percentages)
    for 2023 and 2024. Raw counts are shown as text on the bars.
    """
    grades = ['7','6','5','4','3','2','1','P','N']
    
    # Get row for subject from each year – if missing, create a row with 0 values.
    row2023 = data2023[data2023["SUBJECT"] == subject]
    row2024 = data2024[data2024["SUBJECT"] == subject]
    
    if row2023.empty:
        counts2023 = [0]*len(grades)
        entries2023 = 0
    else:
        counts2023 = [row2023.iloc[0][g] for g in grades]
        entries2023 = row2023.iloc[0]["ENTRIES"]
    
    if row2024.empty:
        counts2024 = [0]*len(grades)
        entries2024 = 0
    else:
        counts2024 = [row2024.iloc[0][g] for g in grades]
        entries2024 = row2024.iloc[0]["ENTRIES"]
    
    # Calculate percentages (guard against division by zero)
    perc2023 = [ (c/entries2023*100 if entries2023 > 0 else 0) for c in counts2023 ]
    perc2024 = [ (c/entries2024*100 if entries2024 > 0 else 0) for c in counts2024 ]
    
    # Create figure
    fig = go.Figure(data=[
        go.Bar(
            name="2023",
            x=grades,
            y=perc2023,
            text=[f"{int(c)} ({p:.1f}%)" for c, p in zip(counts2023, perc2023)],
            textposition="auto"
        ),
        go.Bar(
            name="2024",
            x=grades,
            y=perc2024,
            text=[f"{int(c)} ({p:.1f}%)" for c, p in zip(counts2024, perc2024)],
            textposition="auto"
        )
    ])
    
    # Some styling
    fig.update_layout(
        title=f"Grade Distribution for {subject}",
        xaxis_title="Grade",
        yaxis_title="Percentage (%)",
        barmode="group"
    )
    return fig

def plot_average_score_comparison():
    """
    Create a grouped bar chart showing school and worldwide average scores across subjects
    for both 2023 and 2024.
    Only subjects with complete average data are used.
    """
    # Merge the average data on SUBJECT (inner join)
    df_merge = pd.merge(data2023_avg[["SUBJECT", "Average (School)", "Average (Worldwide)"]],
                        data2024_avg[["SUBJECT", "Average (School)", "Average (Worldwide)"]],
                        on="SUBJECT", suffixes=(" 2023", " 2024"))
    
    # Melt data for plotting
    df_long = pd.melt(df_merge, id_vars="SUBJECT", 
                      value_vars=["Average (School) 2023", "Average (Worldwide) 2023", 
                                  "Average (School) 2024", "Average (Worldwide) 2024"],
                      var_name="Category", value_name="Average")
    
    fig = px.bar(df_long, x="SUBJECT", y="Average", color="Category", barmode="group",
                 title="School and Worldwide Average Scores Comparison")
    fig.update_layout(xaxis_title="Subject", yaxis_title="Average Score")
    return fig

def plot_subject_performance_trends():
    """
    Create a heatmap showing the change in performance (both absolute difference and percentage change)
    between 2023 and 2024 for each subject, for both school and worldwide averages.
    Only subjects with complete data in both years are included.
    """
    # Merge average data on SUBJECT
    df_merge = pd.merge(data2023_avg[["SUBJECT", "Average (School)", "Average (Worldwide)"]],
                        data2024_avg[["SUBJECT", "Average (School)", "Average (Worldwide)"]],
                        on="SUBJECT", suffixes=(" 2023", " 2024"))
    
    # Some debugging stuff
    print("Column names of df_merge:")
    print(df_merge.columns)
    
    print("Shapes of data2023_avg and data2024_avg:")
    print(data2023_avg.shape)
    print(data2024_avg.shape)
    # Compute differences and percentage changes
    df_merge["School Diff"] = df_merge["Average (School) 2024"] - df_merge["Average (School) 2023"]
    df_merge["Worldwide Diff"] = df_merge["Average (Worldwide) 2024"] - df_merge["Average (Worldwide) 2023"]
    df_merge["School % Change"] = (df_merge["School Diff"] / df_merge["Average (School) 2023"])*100
    df_merge["Worldwide % Change"] = (df_merge["Worldwide Diff"] / df_merge["Average (Worldwide) 2023"])*100
    
    # Prepare data for heatmap: rows=subjects, columns=metrics
    heat_df = df_merge.set_index("SUBJECT")[["School Diff", "School % Change", "Worldwide Diff", "Worldwide % Change"]]
    
    # Create heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=heat_df.values,
        x=heat_df.columns,
        y=heat_df.index,
        colourscale="RdYlGn",
        colourbar=dict(title="Change")
    ))
    fig.update_layout(title="Subject Performance Trends (2023 vs 2024)")
    return fig

def plot_total_grade_breakdown():
    """
    Create a stacked bar chart showing the overall (aggregated) grade distribution for 2023 and 2024.
    Percentages are used for the bar heights, and the raw counts are included as text labels.
    """
    grades = ['7','6','5','4','3','2','1','P','N']
    
    # Aggregate for each year
    agg2023 = data2023[grades].sum()
    total2023 = data2023['ENTRIES'].sum()
    agg2024 = data2024[grades].sum()
    total2024 = data2024['ENTRIES'].sum()
    
    perc2023 = [ (agg2023[g]/total2023*100 if total2023>0 else 0) for g in grades ]
    perc2024 = [ (agg2024[g]/total2024*100 if total2024>0 else 0) for g in grades ]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="2023",
        x=grades,
        y=perc2023,
        text=[f"{int(agg2023[g])} ({p:.1f}%)" for g, p in zip(grades, perc2023)],
        textposition="inside"
    ))
    fig.add_trace(go.Bar(
        name="2024",
        x=grades,
        y=perc2024,
        text=[f"{int(agg2024[g])} ({p:.1f}%)" for g, p in zip(grades, perc2024)],
        textposition="inside"
    ))
    fig.update_layout(
        title="Total Grade Breakdown (Aggregated)",
        xaxis_title="Grade",
        yaxis_title="Percentage (%)",
        barmode="stack"
    )
    return fig

def plot_top_bottom_performing_subjects():
    """
    Create a ranked bar chart showing the difference in school average scores (2024 minus 2023)
    for each subject (only those with complete data in both years). Bars are coloured green if there is
    improvement and red if there is a decline. Each bar is labeled with both the absolute difference
    and the percentage change.
    """
    # Merge average data on SUBJECT
    df_merge = pd.merge(data2023_avg[["SUBJECT", "Average (School)"]],
                        data2024_avg[["SUBJECT", "Average (School)"]],
                        on="SUBJECT", suffixes=(" 2023", " 2024"))
    
    # Compute differences and percentage changes
    df_merge["Diff"] = df_merge["Average (School) 2024"] - df_merge["Average (School) 2023"]
    df_merge["% Change"] = (df_merge["Diff"] / df_merge["Average (School) 2023"]) * 100
    
    # Sort by difference
    df_merge.sort_values("Diff", inplace=True)
    
    colours = ['red' if diff < 0 else 'green' for diff in df_merge["Diff"]]
    
    fig = go.Figure(go.Bar(
        x=df_merge["Diff"],
        y=df_merge["SUBJECT"],
        orientation='h',
        marker_colour=colours,
        text=[f"{diff:.2f} ({pct:.1f}%)" for diff, pct in zip(df_merge["Diff"], df_merge["% Change"])],
        textposition="auto"
    ))
    fig.update_layout(
        title="Change in School Average Scores by Subject (2024 vs 2023)",
        xaxis_title="Difference in Average Score",
        yaxis_title="Subject",
        margin=dict(l=150)
    )
    return fig

def plot_specific_subject_comparison(compare_type, subject1, subject2=None, year=None):
    """
    Depending on the compare_type:
      - "Two Subjects in Same Year": Compare the grade distributions and averages for two subjects for a given year.
      - "Same Subject in Different Years": Compare the grade distribution and averages for one subject across 2023 and 2024.
    """
    grades = ['7','6','5','4','3','2','1','P','N']
    
    if compare_type == "Two Subjects in Same Year":
        # year is provided (e.g., "2023" or "2024")
        if year == "2023":
            df = data2023
        else:
            df = data2024
            
        # Get rows for each subject; if missing, use zeros.
        row1 = df[df["SUBJECT"] == subject1]
        row2 = df[df["SUBJECT"] == subject2]
        if row1.empty:
            counts1 = [0]*len(grades)
            entries1 = 0
            avg1 = np.nan
        else:
            counts1 = [row1.iloc[0][g] for g in grades]
            entries1 = row1.iloc[0]["ENTRIES"]
            avg1 = row1.iloc[0]["Average (School)"]
        if row2.empty:
            counts2 = [0]*len(grades)
            entries2 = 0
            avg2 = np.nan
        else:
            counts2 = [row2.iloc[0][g] for g in grades]
            entries2 = row2.iloc[0]["ENTRIES"]
            avg2 = row2.iloc[0]["Average (School)"]
            
        # Calculate percentages
        perc1 = [ (c/entries1*100 if entries1>0 else 0) for c in counts1 ]
        perc2 = [ (c/entries2*100 if entries2>0 else 0) for c in counts2 ]
        
        # Create subplots: two columns
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=[f"{subject1} (Avg: {avg1:.2f})", f"{subject2} (Avg: {avg2:.2f})"])
        
        fig.add_trace(go.Bar(
            x=grades,
            y=perc1,
            text=[f"{int(c)} ({p:.1f}%)" for c, p in zip(counts1, perc1)],
            textposition="auto",
            name=subject1
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=grades,
            y=perc2,
            text=[f"{int(c)} ({p:.1f}%)" for c, p in zip(counts2, perc2)],
            textposition="auto",
            name=subject2
        ), row=1, col=2)
        
        fig.update_layout(title=f"Grade Distribution Comparison for {subject1} and {subject2} in {year}",
                          yaxis_title="Percentage (%)", barmode="group")
        return fig
    
    elif compare_type == "Same Subject in Different Years":
        # Compare one subject in 2023 vs 2024
        fig = go.Figure()
        for yr, df in zip(["2023", "2024"], [data2023, data2024]):
            row = df[df["SUBJECT"] == subject1]
            if row.empty:
                counts = [0]*len(grades)
                entries = 0
                avg_val = np.nan
            else:
                counts = [row.iloc[0][g] for g in grades]
                entries = row.iloc[0]["ENTRIES"]
                avg_val = row.iloc[0]["Average (School)"]
            perc = [ (c/entries*100 if entries>0 else 0) for c in counts ]
            fig.add_trace(go.Bar(
                name=yr,
                x=grades,
                y=perc,
                text=[f"{int(c)} ({p:.1f}%)" for c, p in zip(counts, perc)],
                textposition="auto"
            ))
        fig.update_layout(
            title=f"Grade Distribution for {subject1}: 2023 vs 2024",
            xaxis_title="Grade",
            yaxis_title="Percentage (%)",
            barmode="group"
        )
        return fig

def plot_school_vs_worldwide_comparison(year):
    """
    Create a ranked bar chart showing the percentage difference between school and worldwide averages
    for each subject, for the selected year.
    """
    # This was so annoying to make
    # Select the appropriate dataset based on the year
    if year == "2023":
        df = data2023_avg.copy()
    else:
        df = data2024_avg.copy()
    
    # Calculate the percentage difference
    df.loc[:, "% Difference (School vs Worldwide)"] = ((df["Average (School)"] - df["Average (Worldwide)"]) / df["Average (Worldwide)"]) * 100
    
    # Sort subjects by percentage difference
    df_sorted = df.sort_values("% Difference (School vs Worldwide)")

    # Set colours: green for positive, red for negative differences
    colours = ['red' if diff < 0 else 'green' for diff in df_sorted["% Difference (School vs Worldwide)"]]
    
    # Plot the bar chart
    fig = go.Figure(go.Bar(
        x=df_sorted["% Difference (School vs Worldwide)"],
        y=df_sorted["SUBJECT"],
        orientation='h',
        marker_colour=colours,
        text=[f"{diff:.1f}%" for diff in df_sorted["% Difference (School vs Worldwide)"]],
        textposition="auto"
    ))
    fig.update_layout(
        title=f"School vs Worldwide Average Score Difference ({year})",
        xaxis_title="Percentage Difference (%)",
        yaxis_title="Subject",
        margin=dict(l=150)
    )
    
    return fig


# -------------- PyQt6 Dashboard Application --------------

class DashboardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IB Examination Results Dashboard")
        self.resize(1200, 800)
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        # Layouts
        self.main_layout = QVBoxLayout()
        self.controls_layout = QGridLayout()
        self.main_widget.setLayout(self.main_layout)
        
        # Main visualisation dropdown
        self.vis_dropdown = QComboBox()
        self.vis_dropdown.addItems([
            "Grade Distribution by Subject",
            "Average Score Comparison",
            "Subject Performance Trends Heatmap",
            "Total Grade Breakdown",
            "Top/Bottom Performing Subjects",
            "School vs Worldwide Comparison",
            "Specific Subject Comparison"
        ])
        self.controls_layout.addWidget(QLabel("Select visualisation:"), 0, 0)
        self.controls_layout.addWidget(self.vis_dropdown, 0, 1)
        
        # Additional controls for "Grade Distribution by Subject" and "Specific Subject Comparison"
        # For Grade Distribution by Subject: choose a subject.
        self.subject_label = QLabel("Subject:")
        self.subject_dropdown = QComboBox()
        self.subject_dropdown.addItems(subjects_all)
        self.controls_layout.addWidget(self.subject_label, 1, 0)
        self.controls_layout.addWidget(self.subject_dropdown, 1, 1)
        
        # For Specific Subject Comparison: a dropdown for comparison type.
        self.comp_type_label = QLabel("Comparison Type:")
        self.comp_type_dropdown = QComboBox()
        self.comp_type_dropdown.addItems(["Two Subjects in Same Year", "Same Subject in Different Years"])
        self.controls_layout.addWidget(self.comp_type_label, 2, 0)
        self.controls_layout.addWidget(self.comp_type_dropdown, 2, 1)
        
        self.compare_year_label = QLabel("Year:")
        self.compare_year_dropdown = QComboBox()
        self.compare_year_dropdown.addItems(["2023", "2024"])
        self.controls_layout.addWidget(self.compare_year_label, 7, 0)
        self.controls_layout.addWidget(self.compare_year_dropdown, 7, 1)

        # For Two Subjects in Same Year: subject 1, subject 2 and year.
        self.spec_subj1_label = QLabel("Subject 1:")
        self.spec_subj1_dropdown = QComboBox()
        self.spec_subj1_dropdown.addItems(subjects_all)
        self.controls_layout.addWidget(self.spec_subj1_label, 3, 0)
        self.controls_layout.addWidget(self.spec_subj1_dropdown, 3, 1)
        
        self.spec_subj2_label = QLabel("Subject 2:")
        self.spec_subj2_dropdown = QComboBox()
        self.spec_subj2_dropdown.addItems(subjects_all)
        self.controls_layout.addWidget(self.spec_subj2_label, 4, 0)
        self.controls_layout.addWidget(self.spec_subj2_dropdown, 4, 1)
        
        self.year_label = QLabel("Year:")
        self.year_dropdown = QComboBox()
        self.year_dropdown.addItems(["2023", "2024"])
        self.controls_layout.addWidget(self.year_label, 5, 0)
        self.controls_layout.addWidget(self.year_dropdown, 5, 1)
        
        # Process Button
        self.process_button = QPushButton("Generate visualisation")
        self.controls_layout.addWidget(self.process_button, 6, 0, 1, 2)
        self.process_button.clicked.connect(self.generate_visualisation)
        
        self.main_layout.addLayout(self.controls_layout)
        
        # Web view to display Plotly charts
        self.web_view = QWebEngineView()
        self.main_layout.addWidget(self.web_view)
        
        # Initially hide controls not needed for some visualisations.
        self.update_controls_visibility()
        self.vis_dropdown.currentIndexChanged.connect(self.update_controls_visibility)
        self.comp_type_dropdown.currentIndexChanged.connect(self.update_controls_visibility)
    
    def update_controls_visibility(self):
        """
        Show or hide additional input controls based on the current visualisation choice.
        """
        current_vis = self.vis_dropdown.currentText()
        if current_vis == "Grade Distribution by Subject":
            self.subject_label.show()
            self.subject_dropdown.show()
            self.comp_type_label.hide()
            self.comp_type_dropdown.hide()
            self.spec_subj1_label.hide()
            self.spec_subj1_dropdown.hide()
            self.spec_subj2_label.hide()
            self.spec_subj2_dropdown.hide()
            self.year_label.hide()
            self.year_dropdown.hide()
        elif current_vis == "Specific Subject Comparison":
            # Show the comparison type dropdown.
            self.comp_type_label.show()
            self.comp_type_dropdown.show()
            # Depending on comparison type, show extra fields.
            comp_type = self.comp_type_dropdown.currentText()
            if comp_type == "Two Subjects in Same Year":
                self.spec_subj1_label.show()
                self.spec_subj1_dropdown.show()
                self.spec_subj2_label.show()
                self.spec_subj2_dropdown.show()
                self.year_label.show()
                self.year_dropdown.show()
                self.subject_label.hide()
                self.subject_dropdown.hide()
            else:  # "Same Subject in Different Years"
                self.spec_subj1_label.setText("Subject:")
                self.spec_subj1_label.show()
                self.spec_subj1_dropdown.show()
                self.spec_subj2_label.hide()
                self.spec_subj2_dropdown.hide()
                self.year_label.hide()
                self.year_dropdown.hide()
                self.subject_label.hide()
                self.subject_dropdown.hide()
        elif current_vis == "School vs Worldwide Comparison": ## new added
            self.compare_year_label.show()
            self.compare_year_dropdown.show()
            self.subject_label.hide()
            self.subject_dropdown.hide()
            self.comp_type_label.hide()
            self.comp_type_dropdown.hide()
            self.spec_subj1_label.hide()
            self.spec_subj1_dropdown.hide()
            self.spec_subj2_label.hide()
            self.spec_subj2_dropdown.hide()
            self.year_label.hide()
            self.year_dropdown.hide()
        else:
            # For other visualisations, hide extra controls.
            self.subject_label.hide()
            self.subject_dropdown.hide()
            self.comp_type_label.hide()
            self.comp_type_dropdown.hide()
            self.spec_subj1_label.hide()
            self.spec_subj1_dropdown.hide()
            self.spec_subj2_label.hide()
            self.spec_subj2_dropdown.hide()
            self.year_label.hide()
            self.year_dropdown.hide()
            # new added
            self.compare_year_label.hide()
            self.compare_year_dropdown.hide()
    
    def generate_visualisation(self):
        """
        Called when the "Generate visualisation" button is clicked.
        Creates the appropriate Plotly figure based on the selected visualisation and inputs,
        then renders it in the QWebEngineView.
        """
        current_vis = self.vis_dropdown.currentText()
        fig = None
        
        if current_vis == "Grade Distribution by Subject":
            subject = self.subject_dropdown.currentText()
            fig = plot_grade_distribution(subject)
        
        elif current_vis == "Average Score Comparison":
            fig = plot_average_score_comparison()
            
        elif current_vis == "Subject Performance Trends Heatmap":
            fig = plot_subject_performance_trends()
            
        elif current_vis == "Total Grade Breakdown":
            fig = plot_total_grade_breakdown()
            
        elif current_vis == "Top/Bottom Performing Subjects":
            fig = plot_top_bottom_performing_subjects()

        elif current_vis == "School vs Worldwide Comparison":
            year = self.compare_year_dropdown.currentText()
            fig = plot_school_vs_worldwide_comparison(year)
            
        elif current_vis == "Specific Subject Comparison":
            comp_type = self.comp_type_dropdown.currentText()
            if comp_type == "Two Subjects in Same Year":
                subj1 = self.spec_subj1_dropdown.currentText()
                subj2 = self.spec_subj2_dropdown.currentText()
                year = self.year_dropdown.currentText()
                fig = plot_specific_subject_comparison(comp_type, subj1, subject2=subj2, year=year)
            else:  # "Same Subject in Different Years"
                subject = self.spec_subj1_dropdown.currentText()
                fig = plot_specific_subject_comparison(comp_type, subject)
        
        if fig is not None:
            # Render the Plotly figure as HTML and set it in the web view.
            html = fig.to_html(include_plotlyjs='cdn')
            self.web_view.setHtml(html)

# -------------- Main Application --------------

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DashboardApp()
    window.show()
    sys.exit(app.exec())
