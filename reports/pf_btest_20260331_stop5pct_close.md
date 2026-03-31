======================================================================
QU100 PORTFOLIO BACKTEST — PATTERN-BASED ENTRY/EXIT
======================================================================

  Period:         2022-05-25 to 2026-03-30
  Start capital:  $100.00
  Final capital:  $269.73
  Max positions:  5 (20% each)
  Entry:          Top 2 by pattern confidence
  Patterns:       false_breakdown, false_breakdown_w_bottom
  Hard stop:      5% max loss
  Price mode:     close price only

PERFORMANCE
----------------------------------------------------------------------
  Total trades:       170
  Win rate:           42.9%
  Avg return/trade:   +3.64%
  Median return:      -5.21%
  Total return:       +169.73%
  Sharpe ratio:       1.34
  Max drawdown:       22.18%

  SPY benchmark:      +70.24%
  Alpha:              +99.50%

EXIT REASONS
----------------------------------------------------------------------
  Stop loss:          95
  Target hit:         70
  Max hold:           0
  Pattern invalid:    0
  End of backtest:    5

TOP 5 TRADES
----------------------------------------------------------------------
  Symbol   Pattern                         Entry     Exit   Return      PnL Reason              
  U        false_breakdown_w_bottom     $  19.08 $  43.82 +129.7% $ +35.52 target_hit          
  DJT      false_breakdown_w_bottom     $  24.12 $  51.51 +113.6% $ +28.73 target_hit          
  FAS      false_breakdown_w_bottom     $  98.40 $ 182.54  +85.5% $ +21.93 target_hit          
  AMD      false_breakdown_w_bottom     $ 128.24 $ 203.71  +58.9% $ +18.62 target_hit          
  INTC     false_breakdown_w_bottom     $  24.33 $  42.78  +75.8% $ +16.96 target_hit          

WORST 5 TRADES
----------------------------------------------------------------------
  COIN     false_breakdown_w_bottom     $ 324.24 $ 284.72  -12.2% $  -3.86 stop_loss           
  ROKU     false_breakdown_w_bottom     $  77.51 $  65.70  -15.2% $  -4.67 stop_loss           
  PPG      false_breakdown_w_bottom     $ 123.54 $ 111.87   -9.4% $  -4.87 stop_loss           
  BILI     false_breakdown_w_bottom     $  22.00 $  18.30  -16.8% $  -5.07 stop_loss           
  SQQQ     false_breakdown_w_bottom     $ 200.53 $ 161.63  -19.4% $  -5.51 stop_loss           
======================================================================