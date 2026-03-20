import pandas as pd
import json

with open('arx_model.json') as f:
    config = json.load(f)

print("Validation metrics:", config['metrics_val'])
