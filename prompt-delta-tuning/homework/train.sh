#! /bin/bash

cd $(dirname $0)
cd ..
python -u homework/optqa.py --plm_path ./plm_cache/opt-350m |& tee homework/350m.log
