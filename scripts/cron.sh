#!/bin/bash
/usr/bin/python /workspace/202205_idx-trading/scripts/cron.py
git add /workspace/202205_idx-trading/_benchmarks/paper_trading/
git add /workspace/202205_idx-trading/_data/
git add /workspace/202205_idx-trading/_logs/cron.log
git add /workspace/202205_idx-trading/_metadata/data.csv
git add /workspace/202205_idx-trading/_metadata/paper_trade.csv
git commit -m "Cron for Paper Trading and Data - $(date '+%Y-%m-%d')"
git push origin master
