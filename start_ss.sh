#!/bin/bash
source /Users/sagnikdas/GitHub/Data-Streaming-Pipeline/.venv/bin/activate
export FLASK_APP=superset                     # Set the Flask app variable
superset run -p 8088                          # Start Superset on port 8088
