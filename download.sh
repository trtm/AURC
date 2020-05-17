#!/bin/sh

pip install -r requirements.txt

python3 src/download.py -w=0.1 --topic=abortion
python3 src/download.py -w=0.1 --topic=cloning
python3 src/download.py -w=0.1 --topic=death\ penalty
python3 src/download.py -w=0.1 --topic=gun\ control
python3 src/download.py -w=0.1 --topic=marijuana\ legalization
python3 src/download.py -w=0.1 --topic=minimum\ wage
python3 src/download.py -w=0.1 --topic=nuclear\ energy
python3 src/download.py -w=0.1 --topic=school\ uniforms
