======================================================================
QU100 PORTFOLIO BACKTEST — PATTERN-BASED ENTRY/EXIT
======================================================================

  Period:         2022-05-25 to 2026-03-30
  Start capital:  $100.00
  Final capital:  $209.74
  Max positions:  5 (20% each)
  Entry:          Top 2 by pattern confidence
  Patterns:       false_breakdown, false_breakdown_w_bottom
  Hard stop:      7% max loss
  Price mode:     close + stop-limit order

PERFORMANCE
----------------------------------------------------------------------
  Total trades:       140
  Win rate:           41.4%
  Avg return/trade:   +3.25%
  Median return:      -7.00%
  Total return:       +109.74%
  Sharpe ratio:       1.12
  Max drawdown:       18.89%

  SPY benchmark:      +70.24%
  Alpha:              +39.51%

EXIT REASONS
----------------------------------------------------------------------
  Stop loss:          79
  Target hit:         56
  Max hold:           0
  Pattern invalid:    0
  End of backtest:    5

TOP 5 TRADES
----------------------------------------------------------------------
  Symbol   Pattern                         Entry     Exit   Return      PnL Reason              
  U        false_breakdown_w_bottom     $  19.08 $  43.82 +129.7% $ +29.67 target_hit          
  AMD      false_breakdown_w_bottom     $ 128.24 $ 203.71  +58.9% $ +15.35 target_hit          
  INTC     false_breakdown_w_bottom     $  25.83 $  42.78  +65.6% $ +13.16 target_hit          
  SBUX     false_breakdown_w_bottom     $  71.99 $ 110.00  +52.8% $ +12.70 target_hit          
  USO      false_breakdown_w_bottom     $  69.30 $  91.56  +32.1% $ +12.37 target_hit          

WORST 5 TRADES
----------------------------------------------------------------------
  TGT      false_breakdown_w_bottom     $  89.53 $  83.26   -7.0% $  -2.71 stop_loss           
  ACN      false_breakdown_w_bottom     $ 280.96 $ 261.29   -7.0% $  -2.81 stop_loss           
  PPG      false_breakdown_w_bottom     $ 123.54 $ 114.89   -7.0% $  -2.95 stop_loss           
  BKNG     false_breakdown_w_bottom     $4302.62 $4001.44   -7.0% $  -2.97 stop_loss           
  CRWD     false_breakdown_w_bottom     $ 413.31 $ 384.38   -7.0% $  -2.98 stop_loss           
======================================================================