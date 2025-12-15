#!/usr/bin/env python3
"""
Test script to run CFD simulations with output redirection.
"""
import os
import sys

# Redirect output to file
output_file = '/Users/nreef/Desktop/CFD_Final_Project/cfd_simulation.log'
sys.stdout = open(output_file, 'w')
sys.stderr = sys.stdout

print("Starting CFD simulation...")
print(f"Output file: {output_file}")
print()

# Change to project directory
os.chdir('/Users/nreef/Desktop/CFD_Final_Project')

# Import and run main
from CFDSolver.main import main

# Simulate command-line args
sys.argv = [
    'main.py',
    '--Re', '100', '400', '1000', '3200',
    '--grids', '32', '65', '129',
    '--max-steps', '60000',
    '--tol', '1e-6',
    '--save'
]

try:
    main()
    print("\n\nSimulation completed successfully!")
except Exception as e:
    print(f"\n\nError occurred: {e}")
    import traceback
    traceback.print_exc()

sys.stdout.close()
