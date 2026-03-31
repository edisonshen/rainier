======================================================================
QU100 PORTFOLIO BACKTEST — PATTERN-BASED ENTRY/EXIT
======================================================================

  Period:         2022-05-25 to 2026-03-30
  Start capital:  $100.00
  Final capital:  $225.86
  Max positions:  5 (20% each)
  Entry:          Top 2 by pattern confidence
  Patterns:       false_breakdown, false_breakdown_w_bottom
  Hard stop:      11% max loss
  Price mode:     close + stop-limit order

PERFORMANCE
----------------------------------------------------------------------
  Total trades:       123
  Win rate:           51.2%
  Avg return/trade:   +4.24%
  Median return:      +1.45%
  Total return:       +125.86%
  Sharpe ratio:       1.13
  Max drawdown:       22.36%

  SPY benchmark:      +70.24%
  Alpha:              +55.62%

EXIT REASONS
----------------------------------------------------------------------
  Stop loss:          58
  Target hit:         60
  Max hold:           0
  Pattern invalid:    0
  End of backtest:    5

TOP 5 TRADES
----------------------------------------------------------------------
  Symbol   Pattern                         Entry     Exit   Return      PnL Reason              
  U        false_breakdown_w_bottom     $  19.08 $  43.82 +129.7% $ +32.59 target_hit          
  AMD      false_breakdown_w_bottom     $ 128.24 $ 203.71  +58.9% $ +17.01 target_hit          
  TGT      false_breakdown_w_bottom     $  89.53 $ 118.78  +32.7% $ +13.83 end_of_backtest     
  BHC      false_breakdown_w_bottom     $   5.95 $  10.49  +76.3% $ +13.32 target_hit          
  ALNY     false_breakdown_w_bottom     $ 165.70 $ 245.17  +48.0% $ +11.71 target_hit          

WORST 5 TRADES
----------------------------------------------------------------------
  EH       false_breakdown_w_bottom     $  14.17 $  12.61  -11.0% $  -4.61 stop_loss           
  NKE      false_breakdown_w_bottom     $  61.77 $  54.98  -11.0% $  -4.91 stop_loss           
  ZS       false_breakdown_w_bottom     $ 169.39 $ 150.76  -11.0% $  -4.93 stop_loss           
  ACN      false_breakdown_w_bottom     $ 280.96 $ 250.05  -11.0% $  -4.95 stop_loss           
  BKNG     false_breakdown_w_bottom     $4302.62 $3829.34  -11.0% $  -5.06 stop_loss           
======================================================================