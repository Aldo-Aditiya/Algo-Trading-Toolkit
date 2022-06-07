from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional


app = FastAPI()

@app.get("/pair_trade_signal")
def function(ticker_pair: List[str]):
    
    no_lots = 1
    
    response = {
                    'signal' = signal,
                    'no_lots' = no_lots,
                    'price' = price,
                    'ticker' = ticker
                }
    return response