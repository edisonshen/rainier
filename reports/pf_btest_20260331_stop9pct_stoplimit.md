======================================================================
QU100 PORTFOLIO BACKTEST — PATTERN-BASED ENTRY/EXIT
======================================================================

  Period:         2022-05-25 to 2026-03-30
  Start capital:  $100.00
  Final capital:  $283.09
  Max positions:  5 (20% each)
  Entry:          Top 2 by pattern confidence
  Patterns:       false_breakdown, false_breakdown_w_bottom
  Hard stop:      9% max loss
  Price mode:     close + stop-limit order

PERFORMANCE
----------------------------------------------------------------------
  Total trades:       120
  Win rate:           52.5%
  Avg return/trade:   +5.44%
  Median return:      +3.00%
  Total return:       +183.09%
  Sharpe ratio:       1.51
  Max drawdown:       16.94%

  SPY benchmark:      +70.24%
  Alpha:              +112.85%

EXIT REASONS
----------------------------------------------------------------------
  Stop loss:          55
  Target hit:         60
  Max hold:           0
  Pattern invalid:    0
  End of backtest:    5

TOP 5 TRADES
----------------------------------------------------------------------
  Symbol   Pattern                         Entry     Exit   Return      PnL Reason              
  U        false_breakdown_w_bottom     $  19.08 $  43.82 +129.7% $ +39.93 target_hit          
  AMD      false_breakdown_w_bottom     $ 128.24 $ 203.71  +58.9% $ +20.83 target_hit          
  TGT      false_breakdown_w_bottom     $  89.53 $ 118.78  +32.7% $ +16.78 end_of_backtest     
  BHC      false_breakdown_w_bottom     $   5.95 $  10.49  +76.3% $ +16.53 target_hit          
  INTC     false_breakdown_w_bottom     $  25.66 $  42.78  +66.7% $ +13.71 target_hit          

WORST 5 TRADES
----------------------------------------------------------------------
  CRWV     false_breakdown_w_bottom     $ 117.14 $ 106.60   -9.0% $  -4.09 stop_loss           
  EH       false_breakdown_w_bottom     $  14.17 $  12.89   -9.0% $  -4.63 stop_loss           
  ACN      false_breakdown_w_bottom     $ 280.96 $ 255.67   -9.0% $  -4.97 stop_loss           
  PPG      false_breakdown_w_bottom     $ 123.54 $ 112.42   -9.0% $  -5.00 stop_loss           
  BKNG     false_breakdown_w_bottom     $4302.62 $3915.39   -9.0% $  -5.22 stop_loss           
======================================================================