from pathlib import Path
from typing import Callable, Optional

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from textwrap import wrap
from src.config import REPORTS_DIR, PLOTS_DIR


ArtifactLogger = Optional[Callable[[str], None]]

def draw_wrapped_text(c, text, x, y, max_width, leading=14):
    for line in wrap(text, width=int(max_width / 6)):
        c.drawString(x, y, line)
        y -= leading
    return y

def generate_report(artifact_logger: ArtifactLogger = None):
    pdf_path = REPORTS_DIR / "baseline_task_stream_A_report.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=LETTER)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 720, "Task Stream A – Data & Visualization Lead")
    c.setFont("Helvetica", 10)
    c.drawString(72, 700, "Authors: Wanhar Aziz & Jonathan Zul Luna")

    text = (
        "This report summarizes preprocessing and exploratory data analysis for the "
        "Personal Finance ML Dataset. Missing values in 'loan_type' were handled by adding "
        "a 'Missing' category. Deterministic train/val/test splits were created using a fixed "
        "random seed. Two EDA plots—class distribution and correlation heatmap—are presented below."
    )
    y = draw_wrapped_text(c, text, 72, 660, 480)

    class_plot = PLOTS_DIR / "class_distribution_has_loan.png"
    if class_plot.exists():
        c.drawImage(str(class_plot), 72, 360, width=400, height=250)
    else:
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(72, 450, "Class distribution plot not available.")
        c.setFont("Helvetica", 10)

    heatmap_plot = PLOTS_DIR / "correlation_heatmap.png"
    if heatmap_plot.exists():
        c.drawImage(str(heatmap_plot), 72, 60, width=400, height=250)
    else:
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(72, 150, "Correlation heatmap not available.")
        c.setFont("Helvetica", 10)

    c.save()

    if artifact_logger is not None:
        artifact_logger(str(pdf_path))
    print(f"Report generated: {pdf_path}")
