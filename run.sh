#!/bin/bash
set -euo pipefail 
trap 'kill $(jobs -p) 2>/dev/null' EXIT 

# python main.py --config ./config/InceptionTimespo2.json
# python main.py --config ./config/Resnetspo2.json
# python main.py --config ./config/Mamba2spo2.json

# python main.py --config ./config/Resnetbp.json
# python main.py --config ./config/InceptionTimebp.json
# python main.py --config ./config/Mamba2bp.json


# python main.py --config ./config/InceptionTimespo2-2.json
# python main.py --config ./config/Resnetspo2-2.json
# python main.py --config ./config/Mamba2spo2-2.json

# python main.py --config ./config/Resnetbp-2.json
# python main.py --config ./config/InceptionTimebp-2.json
# python main.py --config ./config/Mamba2bp-2.json

python main.py --config ./config/Transformerbp.json
python main.py --config ./config/Transformerspo2.json

python main.py --config ./config/Transformerbp-2.json
python main.py --config ./config/Transformerspo2-2.json