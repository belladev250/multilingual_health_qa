import os
import json
import re

def convert_py_to_ipynb(py_path: str, ipynb_path: str):
    """
    Converts a Python file structured with '# %%' cell delimiters 
    into a standard Jupyter Notebook (.ipynb) file.
    """
    with open(py_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    cells = []
    current_cell_type = 'code'
    current_lines = []
    
    def add_cell(cell_type, lines_list):
        if not lines_list:
            return
        
        # Clean lines
        cleaned_lines = []
        if cell_type == 'markdown':
            for line in lines_list:
                # Strip leading '# ' or '#' from markdown cells
                stripped = re.sub(r'^#\s?', '', line)
                cleaned_lines.append(stripped)
        else:
            cleaned_lines = lines_list
            
        # Strip trailing newlines from the last elements
        if cleaned_lines and cleaned_lines[-1] == '\n':
            cleaned_lines.pop()
            
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": cleaned_lines
        }
        if cell_type == 'code':
            cell["execution_count"] = None
            cell["outputs"] = []
            
        cells.append(cell)

    for line in lines:
        match = re.match(r'^#\s*%%\s*(.*)', line)
        if match:
            # Save the current active cell before starting a new one
            add_cell(current_cell_type, current_lines)
            
            # Determine new cell type
            cell_arg = match.group(1).strip()
            if 'markdown' in cell_arg:
                current_cell_type = 'markdown'
            else:
                current_cell_type = 'code'
                
            current_lines = []
        else:
            current_lines.append(line)
            
    # Add the final cell
    add_cell(current_cell_type, current_lines)
    
    # Construct complete notebook JSON structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print(f"Successfully converted {os.path.basename(py_path)} -> {os.path.basename(ipynb_path)}")

if __name__ == "__main__":
    notebooks_dir = "/Users/pixeleyeblue/.gemini/antigravity/scratch/multilingual_health_qa/notebooks"
    
    py_files = [
        "01_exploratory_data_analysis.py",
        "02_preprocessing_and_tokenization.py",
        "03_training_and_fine_tuning.py"
    ]
    
    for filename in py_files:
        py_path = os.path.join(notebooks_dir, filename)
        ipynb_filename = filename.replace(".py", ".ipynb")
        ipynb_path = os.path.join(notebooks_dir, ipynb_filename)
        
        if os.path.exists(py_path):
            convert_py_to_ipynb(py_path, ipynb_path)
        else:
            print(f"Warning: {filename} not found.")
