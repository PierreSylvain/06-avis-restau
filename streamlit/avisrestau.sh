#!/bin/bash 
cd /home/ubuntu/avisrestau
source ./venv/bin/activate
export PATH=$PATH:/home/ubuntu/.local/bin
nohup streamlit run dashboard.py
