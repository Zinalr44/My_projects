#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nbformat
from nbconvert import PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor

# Load the notebook
with open('assignment.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

# Create a preprocessor
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

# Execute the notebook
ep.preprocess(nb, {'metadata': {'path': './'}})

# Export to script
exporter = PythonExporter()
body, _ = exporter.from_notebook_node(nb)

# Save the script
with open('assignment.py', 'w') as f:
    f.write(body)

# Run the script
import subprocess
subprocess.run(["python", "assignment.py"])

