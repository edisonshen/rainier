======================================================================
QU100 PORTFOLIO BACKTEST — PATTERN-BASED ENTRY/EXIT
======================================================================

  Period:         2022-05-25 to 2026-03-30
  Start capital:  $100.00
  Final capital:  $249.36
  Max positions:  5 (20% each)
  Entry:          Top 2 by pattern confidence
  Patterns:       false_breakdown, false_breakdown_w_bottom
  Hard stop:      9% max loss
  Price mode:     close price only

PERFORMANCE
----------------------------------------------------------------------
  Total trades:       126
  Win rate:           53.2%
  Avg return/trade:   +4.57%
  Median return:      +2.77%
  Total return:       +149.36%
  Sharpe ratio:       1.21
  Max drawdown:       19.37%

  SPY benchmark:      +70.24%
  Alpha:              +79.13%

EXIT REASONS
----------------------------------------------------------------------
  Stop loss:          57
  Target hit:         64
  Max hold:           0
  Pattern invalid:    0
  End of backtest:    5

TOP 5 TRADES
----------------------------------------------------------------------
  Symbol   Pattern                         Entry     Exit   Return      PnL Reason              
  U        false_breakdown_w_bottom     $  19.08 $  43.82 +129.7% $ +36.48 target_hit          
  AMD      false_breakdown_w_bottom     $ 128.24 $ 203.71  +58.9% $ +18.97 target_hit          
  BHC      false_breakdown_w_bottom     $   5.95 $  10.49  +76.3% $ +15.94 target_hit          
  TGT      false_breakdown_w_bottom     $  89.53 $ 118.78  +32.7% $ +15.70 end_of_backtest     
  ALNY     false_breakdown_w_bottom     $ 165.70 $ 245.17  +48.0% $ +14.78 target_hit          

WORST 5 TRADES
----------------------------------------------------------------------
  ACN      false_breakdown_w_bottom     $ 280.96 $ 241.21  -14.1% $  -6.96 stop_loss           
  ZS       false_breakdown_w_bottom     $ 169.39 $ 143.28  -15.4% $  -7.65 stop_loss           
  SNAP     false_breakdown_w_bottom     $   9.69 $   7.78  -19.7% $  -7.76 stop_loss           
  SQQQ     false_breakdown_w_bottom     $ 237.45 $ 161.63  -31.9% $  -9.31 stop_loss           
  CPRI     false_breakdown_w_bottom     $  35.44 $  20.48  -42.2% $ -13.32 stop_loss           
======================================================================