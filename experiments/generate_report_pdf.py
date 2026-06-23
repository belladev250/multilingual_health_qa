import os
import sys
import re
from fpdf import FPDF

class AcademicPDF(FPDF):
    def header(self):
        # Header only on later pages, not the title page (page 1)
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 110, 120)
            self.cell(0, 8, "Machine Learning Techniques I  --  Final Course Project Report", align="L")
            self.set_x(-60)
            self.cell(0, 8, "Bella Melissa Ineza (June 2026)", align="R")
            self.ln(10)
            self.set_draw_color(220, 225, 230)
            self.line(20, 18, 190, 18)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 110, 120)
        # Centered page number
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_pdf(md_path, pdf_path):
    print(f"Reading markdown report from: {md_path}")
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pdf = AcademicPDF(orientation="P", unit="mm", format="A4")
    # Set margins: Left=20mm, Top=20mm, Right=20mm
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    
    # Text colors
    PRIMARY_COLOR = (30, 41, 59)      # Slate 800 (Charcoal)
    SECONDARY_COLOR = (79, 70, 229)   # Indigo 600
    TEXT_COLOR = (51, 65, 85)         # Slate 700
    MUTED_COLOR = (100, 116, 139)     # Slate 500
    
    pdf.set_text_color(*PRIMARY_COLOR)
    
    in_table = False
    table_headers = []
    table_data = []
    
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        cleaned_line = line.strip()
        
        # 1. Handle Table parsing
        if cleaned_line.startswith("|") and cleaned_line.endswith("|"):
            if not in_table:
                in_table = True
                table_headers = [c.strip() for c in cleaned_line.split("|")[1:-1]]
                table_data = []
                # Skip the next line which is the separator |:---:|---|...
                idx += 2
                continue
            else:
                # Add data row
                row_cells = [c.strip() for c in cleaned_line.split("|")[1:-1]]
                # Remove markdown bold/italics from table cells for clean PDF rendering
                cleaned_cells = []
                for cell in row_cells:
                    cell_clean = re.sub(r"\*\*|\*", "", cell)
                    cell_clean = re.sub(r"`", "", cell_clean)
                    cleaned_cells.append(cell_clean)
                table_data.append(cleaned_cells)
                idx += 1
                continue
        else:
            # If we were in a table and it just ended, render it!
            if in_table:
                in_table = False
                render_table(pdf, table_headers, table_data, SECONDARY_COLOR, PRIMARY_COLOR, TEXT_COLOR)
                pdf.ln(4)
            
        # 2. Skip empty lines
        if not cleaned_line:
            # Let's add a minor paragraph spacing
            pdf.ln(2)
            idx += 1
            continue
            
        # 3. Handle main title (H1)
        if cleaned_line.startswith("# "):
            title = cleaned_line[2:]
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(*SECONDARY_COLOR)
            pdf.multi_cell(0, 8, title, align="C")
            pdf.ln(5)
            idx += 1
            continue
            
        # 4. Handle Subtitle metadata (Author, Course, Date, etc.)
        if cleaned_line.startswith("**Author**:") or cleaned_line.startswith("**Course**:") or cleaned_line.startswith("**Date**:") or cleaned_line.startswith("**IEEE Citation Format**"):
            # Clean formatting bold markers
            meta_text = re.sub(r"\*\*", "", cleaned_line)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*MUTED_COLOR)
            pdf.cell(0, 5, meta_text, align="C")
            pdf.ln(5)
            idx += 1
            continue
            
        if cleaned_line == "---":
            # Decorative horizontal rule
            pdf.ln(3)
            pdf.set_draw_color(226, 232, 240)
            pdf.set_line_width(0.5)
            # draw line across margin width
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(3)
            idx += 1
            continue
            
        # 5. Handle major section headings (H2)
        if cleaned_line.startswith("## "):
            section_title = cleaned_line[3:]
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(*SECONDARY_COLOR)
            pdf.cell(0, 6, section_title)
            pdf.ln(6)
            # Add section divider line
            pdf.set_draw_color(79, 70, 229)
            pdf.set_line_width(0.3)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(2)
            idx += 1
            continue
            
        # 6. Handle subsections (H3)
        if cleaned_line.startswith("### "):
            subsec_title = cleaned_line[4:]
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*PRIMARY_COLOR)
            pdf.cell(0, 5, subsec_title)
            pdf.ln(5)
            idx += 1
            continue
            
        # 7. Handle bullet points
        if cleaned_line.startswith("- ") or cleaned_line.startswith("* "):
            bullet_text = cleaned_line[2:]
            # Standardize bold formatting and encoding
            bullet_text_clean = process_bold_text(bullet_text)
            
            pdf.set_x(25) # indent
            pdf.set_font("Helvetica", "", 9.5)
            pdf.set_text_color(*TEXT_COLOR)
            # Render bullet character
            pdf.cell(4, 5, chr(149))
            
            # Print the text block with markdown bold highlights
            render_inline_styled_text(pdf, bullet_text_clean, TEXT_COLOR)
            pdf.ln(1)
            idx += 1
            continue
            
        # 8. Handle numbered list items (like references or languages list)
        if re.match(r"^\d+\.\s+", cleaned_line) or re.match(r"^\[\d+\]\s+", cleaned_line):
            # References or lists
            text_clean = process_bold_text(cleaned_line)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*TEXT_COLOR)
            render_inline_styled_text(pdf, text_clean, TEXT_COLOR)
            pdf.ln(1)
            idx += 1
            continue

        # 9. Handle mathematical inline or display formulas
        if cleaned_line.startswith("$$") or cleaned_line.endswith("$$"):
            formula = cleaned_line.replace("$$", "").strip()
            if not formula and idx + 1 < len(lines):
                # block formula on next line
                idx += 1
                formula = lines[idx].strip()
                if formula.endswith("$$"):
                    formula = formula[:-2].strip()
            
            pdf.ln(2)
            pdf.set_font("Courier", "B", 9.5)
            pdf.set_text_color(*SECONDARY_COLOR)
            pdf.multi_cell(0, 5, formula, align="C")
            pdf.ln(2)
            idx += 1
            continue
            
        # 10. Handle standard paragraph body text
        paragraph_text = cleaned_line
        paragraph_text_clean = process_bold_text(paragraph_text)
        
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*TEXT_COLOR)
        render_inline_styled_text(pdf, paragraph_text_clean, TEXT_COLOR)
        pdf.ln(1)
        idx += 1

    print(f"Saving beautiful PDF report to: {pdf_path}")
    pdf.output(pdf_path)
    print("PDF Generation complete!")

def process_bold_text(text):
    # Map non-latin-1 specific characters for safe PDF encoding
    text = text.replace("ግዕዝ", "Geez")
    # Replace markdown `code` blocks with normal text for simplicity
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Force encode and decode in latin-1 ignoring bad chars to prevent crashes
    text = text.encode("latin-1", "ignore").decode("latin-1")
    return text

def render_inline_styled_text(pdf, text, default_color):
    """
    Renders text with inline bold formatting '**text**' using MultiCell or standard Write.
    Splits text by '**' and alternates bold/normal styles seamlessly.
    """
    parts = text.split("**")
    is_bold = False
    
    # Store starting x position to handle indentations
    start_x = pdf.get_x()
    
    for part in parts:
        if not part:
            is_bold = not is_bold
            continue
            
        if is_bold:
            pdf.set_font("Helvetica", "B", pdf.font_size_pt)
        else:
            pdf.set_font("Helvetica", "", pdf.font_size_pt)
            
        # Use write to append inline blocks
        pdf.write(5, part)
        is_bold = not is_bold
        
    pdf.ln(5) # Add a line break at the end of the text block
    pdf.set_x(start_x) # Reset x position

def render_table(pdf, headers, rows, header_color, primary_color, text_color):
    """
    Renders a beautifully formatted, compact academic table in the PDF document.
    """
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 8)
    
    # Calculate column widths dynamically based on table size
    num_cols = len(headers)
    if num_cols == 8:
        # Our main results table has 8 columns
        col_widths = [14, 46, 17, 15, 20, 20, 16, 22]
    else:
        # Uniform fallback
        col_widths = [170 / num_cols] * num_cols
        
    # Draw headers
    pdf.set_fill_color(*header_color)
    pdf.set_text_color(255, 255, 255) # white text
    pdf.set_draw_color(226, 232, 240)
    pdf.set_line_width(0.2)
    
    for i, col_name in enumerate(headers):
        align = "C" if i != 1 else "L"
        pdf.cell(col_widths[i], 6, col_name, border=1, align=align, fill=True)
    pdf.ln()
    
    # Draw data rows with zebra striping
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(*text_color)
    
    for row_idx, row in enumerate(rows):
        # Zebra striping
        if row_idx % 2 == 1:
            pdf.set_fill_color(248, 250, 252) # Slate 50
        else:
            pdf.set_fill_color(255, 255, 255) # White
            
        # If it is the final row (EXP-10), make it bold and highlighted
        if len(row) > 0 and "EXP-10" in row[0]:
            pdf.set_font("Helvetica", "B", 7.5)
            pdf.set_fill_color(238, 242, 255) # Light Indigo
            pdf.set_text_color(*primary_color)
        else:
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(*text_color)
            
        for i, cell_val in enumerate(row):
            val_str = str(cell_val)
            align = "C" if i != 1 else "L"
            pdf.cell(col_widths[i], 5.5, val_str, border=1, align=align, fill=True)
        pdf.ln()
        
    # Reset text color and fonts
    pdf.set_text_color(*primary_color)

if __name__ == "__main__":
    project_dir = "/Users/pixeleyeblue/.gemini/antigravity/scratch/multilingual_health_qa"
    md_file = os.path.join(project_dir, "docs/StudentName_FinalProject_Draft.md")
    pdf_file = os.path.join(project_dir, "docs/BellaMelissaIneza_FinalProject.pdf")
    
    if os.path.exists(md_file):
        generate_pdf(md_file, pdf_file)
    else:
        print(f"Error: {md_file} not found.")
