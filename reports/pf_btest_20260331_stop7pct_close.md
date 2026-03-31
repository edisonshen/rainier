======================================================================
QU100 PORTFOLIO BACKTEST — PATTERN-BASED ENTRY/EXIT
======================================================================

  Period:         2022-05-25 to 2026-03-30
  Start capital:  $100.00
  Final capital:  $239.81
  Max positions:  5 (20% each)
  Entry:          Top 2 by pattern confidence
  Patterns:       false_breakdown, false_breakdown_w_bottom
  Hard stop:      7% max loss
  Price mode:     close price only

PERFORMANCE
----------------------------------------------------------------------
  Total trades:       151
  Win rate:           47.0%
  Avg return/trade:   +3.72%
  Median return:      -7.00%
  Total return:       +139.81%
  Sharpe ratio:       1.22
  Max drawdown:       20.64%

  SPY benchmark:      +70.24%
  Alpha:              +69.58%

EXIT REASONS
----------------------------------------------------------------------
  Stop loss:          76
  Target hit:         70
  Max hold:           0
  Pattern invalid:    0
  End of backtest:    5

TOP 5 TRADES
----------------------------------------------------------------------
  Symbol   Pattern                         Entry     Exit   Return      PnL Reason              
  U        false_breakdown_w_bottom     $  19.08 $  43.82 +129.7% $ +35.18 target_hit          
  FAS      false_breakdown_w_bottom     $  98.40 $ 182.54  +85.5% $ +22.23 target_hit          
  AMD      false_breakdown_w_bottom     $ 128.24 $ 203.71  +58.9% $ +18.79 target_hit          
  BHC      false_breakdown_w_bottom     $   5.95 $  10.49  +76.3% $ +17.40 target_hit          
  USO      false_breakdown_w_bottom     $  69.30 $  91.56  +32.1% $ +14.49 target_hit          

WORST 5 TRADES
----------------------------------------------------------------------
  CRWD     false_breakdown_w_bottom     $ 413.31 $ 369.58  -10.6% $  -5.19 stop_loss           
  SQQQ     false_breakdown_w_bottom     $ 200.53 $ 161.63  -19.4% $  -5.34 stop_loss           
  CELH     false_breakdown_w_bottom     $  32.62 $  26.88  -17.6% $  -5.47 stop_loss           
  ZS       false_breakdown_w_bottom     $ 169.39 $ 143.28  -15.4% $  -7.35 stop_loss           
  SNAP     false_breakdown_w_bottom     $   9.69 $   7.78  -19.7% $  -7.79 stop_loss           
======================================================================