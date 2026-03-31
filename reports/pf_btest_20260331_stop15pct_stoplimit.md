======================================================================
QU100 PORTFOLIO BACKTEST — PATTERN-BASED ENTRY/EXIT
======================================================================

  Period:         2022-05-25 to 2026-03-30
  Start capital:  $100.00
  Final capital:  $217.81
  Max positions:  5 (20% each)
  Entry:          Top 2 by pattern confidence
  Patterns:       false_breakdown, false_breakdown_w_bottom
  Hard stop:      15% max loss
  Price mode:     close + stop-limit order

PERFORMANCE
----------------------------------------------------------------------
  Total trades:       84
  Win rate:           63.1%
  Avg return/trade:   +5.21%
  Median return:      +4.30%
  Total return:       +117.81%
  Sharpe ratio:       1.05
  Max drawdown:       21.21%

  SPY benchmark:      +70.24%
  Alpha:              +47.58%

EXIT REASONS
----------------------------------------------------------------------
  Stop loss:          28
  Target hit:         51
  Max hold:           0
  Pattern invalid:    0
  End of backtest:    5

TOP 5 TRADES
----------------------------------------------------------------------
  Symbol   Pattern                         Entry     Exit   Return      PnL Reason              
  AMD      false_breakdown_w_bottom     $ 128.24 $ 203.71  +58.9% $ +22.10 target_hit          
  SNOW     false_breakdown_w_bottom     $ 118.00 $ 171.35  +45.2% $ +14.83 target_hit          
  KWEB     false_breakdown_w_bottom     $  24.53 $  36.30  +48.0% $ +14.69 target_hit          
  INTC     false_breakdown_w_bottom     $  25.66 $  42.78  +66.7% $ +13.43 target_hit          
  UAN      false_breakdown_w_bottom     $  88.57 $ 109.21  +23.3% $ +10.39 target_hit          

WORST 5 TRADES
----------------------------------------------------------------------
  NKE      false_breakdown_w_bottom     $  61.77 $  52.50  -15.0% $  -6.63 stop_loss           
  ZS       false_breakdown_w_bottom     $ 169.39 $ 143.98  -15.0% $  -6.68 stop_loss           
  NIO      false_breakdown_w_bottom     $   6.34 $   5.39  -15.0% $  -6.76 stop_loss           
  EH       false_breakdown_w_bottom     $  14.17 $  12.04  -15.0% $  -7.22 stop_loss           
  ACN      false_breakdown_w_bottom     $ 280.96 $ 238.82  -15.0% $  -7.72 stop_loss           
======================================================================