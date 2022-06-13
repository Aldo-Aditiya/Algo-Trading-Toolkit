#!/bin/bash
python cron.py
git add /workspace/202205_idx-trading/_benchmarks/paper_trading/
git add /workspace/202205_idx-trading/_data/
git commit -m "Cron for Paper Trading and Data - $(date '+%Y-%m-%d')"
git push origin master