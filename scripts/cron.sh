#!/bin/bash
python cron.py
git add /workspace/202205_idx-trading/_benchmarks/paper_trading/
git add /workspace/202205_idx-trading/_data/
DATE 
git commit -m "Cron for Paper Trading and Data - $DATE"