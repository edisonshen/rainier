======================================================================
QU100 PORTFOLIO BACKTEST — PATTERN-BASED ENTRY/EXIT
======================================================================

  Period:         2022-05-25 to 2026-03-30
  Start capital:  $100.00
  Final capital:  $255.75
  Max positions:  5 (20% each)
  Entry:          Top 2 by pattern confidence
  Patterns:       false_breakdown, false_breakdown_w_bottom
  Hard stop:      15% max loss
  Price mode:     close price only

PERFORMANCE
----------------------------------------------------------------------
  Total trades:       83
  Win rate:           65.1%
  Avg return/trade:   +7.03%
  Median return:      +5.32%
  Total return:       +155.75%
  Sharpe ratio:       1.27
  Max drawdown:       27.12%

  SPY benchmark:      +70.24%
  Alpha:              +85.52%

EXIT REASONS
----------------------------------------------------------------------
  Stop loss:          26
  Target hit:         52
  Max hold:           0
  Pattern invalid:    0
  End of backtest:    5

TOP 5 TRADES
----------------------------------------------------------------------
  Symbol   Pattern                         Entry     Exit   Return      PnL Reason              
  U        false_breakdown_w_bottom     $  19.08 $  43.82 +129.7% $ +33.74 target_hit          
  INTC     false_breakdown_w_bottom     $  20.97 $  41.53  +98.0% $ +32.03 target_hit          
  AMD      false_breakdown_w_bottom     $ 128.24 $ 203.71  +58.9% $ +17.83 target_hit          
  TGT      false_breakdown_w_bottom     $  89.53 $ 118.78  +32.7% $ +16.23 end_of_backtest     
  SNOW     false_breakdown_w_bottom     $ 118.00 $ 171.35  +45.2% $ +12.45 target_hit          

WORST 5 TRADES
----------------------------------------------------------------------
  ZS       false_breakdown_w_bottom     $ 169.39 $ 143.28  -15.4% $  -7.71 stop_loss           
  EH       false_breakdown_w_bottom     $  14.17 $  11.86  -16.3% $  -8.01 stop_loss           
  SQQQ     false_breakdown_w_bottom     $ 237.45 $ 161.63  -31.9% $  -8.63 stop_loss           
  ACN      false_breakdown_w_bottom     $ 280.96 $ 233.58  -16.9% $  -8.90 stop_loss           
  CPRI     false_breakdown_w_bottom     $  35.44 $  20.48  -42.2% $ -12.05 stop_loss           
======================================================================