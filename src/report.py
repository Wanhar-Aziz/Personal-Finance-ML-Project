import logging
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

from src.config import REPORTS_DIR, PLOTS_DIR, TABLES_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _build_report_elements() -> list:
    """Builds a list of Platypus Flowables to be rendered in the PDF report."""
    styles = getSampleStyleSheet()
    elements = []

    # --- Header ---
    elements.append(Paragraph("CS-4120 Midpoint Project Report", styles['h1']))
    elements.append(Paragraph("Authors: Wanhar Aziz & Jonathan Zul Luna", styles['h3']))
    elements.append(Spacer(1, 0.25 * inch))

    # --- Introduction ---
    intro_text = (
        "This report summarizes the baseline model performance for the Personal Finance ML Project. "
        "The pipeline includes data cleaning, feature engineering (creating ratio-based features), "
        "and training classical models for two tasks: loan eligibility classification and credit "
        "score regression. All experiments are tracked with MLflow, and the best models are "
        "evaluated on a held-out test set."
    )
    elements.append(Paragraph(intro_text, styles['BodyText']))
    elements.append(Spacer(1, 0.25 * inch))

    # --- Classification Results ---
    elements.append(Paragraph("Task 1: Classification Results", styles['h2']))
    clf_table_path = TABLES_DIR / "classification_test_metrics.csv"
    if clf_table_path.exists():
        df_clf = pd.read_csv(clf_table_path)
        table_data = [df_clf.columns.to_list()] + df_clf.values.tolist()

        # Only 4 decimal places for "value" column
        for i in range(1, len(table_data)):
            table_data[i][1] = f"{table_data[i][1]:.4f}"
            table_data[i][2] = f"{table_data[i][2]:.4f}"

        table = Table(table_data, colWidths=[2 * inch, 1 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
    else:
        elements.append(Paragraph("Classification metrics table not found.", styles['Italic']))
    
    cm_plot_path = PLOTS_DIR / "confusion_matrix.png"
    if cm_plot_path.exists():
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Image(str(cm_plot_path), width=4 * inch, height=3 * inch))
    
    elements.append(Spacer(1, 0.5 * inch))

    # --- Regression Results ---
    elements.append(Paragraph("Task 2: Regression Results", styles['h2']))
    reg_table_path = TABLES_DIR / "regression_test_metrics.csv"
    if reg_table_path.exists():
        df_reg = pd.read_csv(reg_table_path)
        table_data = [df_reg.columns.to_list()] + df_reg.values.tolist()

        # Only 2 decimal places for "value" column
        for i in range(1, len(table_data)):
            table_data[i][1] = f"{table_data[i][1]:.2f}"
            table_data[i][2] = f"{table_data[i][2]:.2f}"
            
        table = Table(table_data, colWidths=[2 * inch, 1.5 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
    else:
        elements.append(Paragraph("Regression metrics table not found.", styles['Italic']))

    residuals_plot_path = PLOTS_DIR / "residuals_plot.png"
    if residuals_plot_path.exists():
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Image(str(residuals_plot_path), width=4 * inch, height=3 * inch))

    return elements

def generate_report() -> None:
    """Generates a PDF report summarizing the pipeline results."""
    output_path = REPORTS_DIR / "midpoint_model_evaluation_report.pdf"
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    
    logging.info("Generating PDF report...")
    report_elements = _build_report_elements()
    doc.build(report_elements)
    logging.info(f"Report generated successfully at {output_path}")

if __name__ == '__main__':
    generate_report()
