======================================================================
QU100 PORTFOLIO BACKTEST — PATTERN-BASED ENTRY/EXIT
======================================================================

  Period:         2022-05-25 to 2026-03-30
  Start capital:  $100.00
  Final capital:  $202.81
  Max positions:  5 (20% each)
  Entry:          Top 2 by pattern confidence
  Patterns:       false_breakdown, false_breakdown_w_bottom
  Hard stop:      5% max loss
  Price mode:     close + stop-limit order

PERFORMANCE
----------------------------------------------------------------------
  Total trades:       223
  Win rate:           31.4%
  Avg return/trade:   +1.94%
  Median return:      -5.00%
  Total return:       +102.81%
  Sharpe ratio:       1.05
  Max drawdown:       30.18%

  SPY benchmark:      +70.24%
  Alpha:              +32.57%

EXIT REASONS
----------------------------------------------------------------------
  Stop loss:          150
  Target hit:         68
  Max hold:           0
  Pattern invalid:    0
  End of backtest:    5

TOP 5 TRADES
----------------------------------------------------------------------
  Symbol   Pattern                         Entry     Exit   Return      PnL Reason              
  DJT      false_breakdown_w_bottom     $  24.12 $  51.51 +113.6% $ +25.18 target_hit          
  AMD      false_breakdown_w_bottom     $ 128.24 $ 203.71  +58.9% $ +17.11 target_hit          
  USO      false_breakdown_w_bottom     $  69.30 $  91.56  +32.1% $ +11.55 target_hit          
  SBUX     false_breakdown_w_bottom     $  71.99 $ 110.00  +52.8% $ +11.54 target_hit          
  INTC     false_breakdown_w_bottom     $  24.33 $  42.78  +75.8% $ +10.61 target_hit          

WORST 5 TRADES
----------------------------------------------------------------------
  ACN      false_breakdown_w_bottom     $ 280.96 $ 266.91   -5.0% $  -1.91 stop_loss           
  ZS       false_breakdown_w_bottom     $ 169.39 $ 160.92   -5.0% $  -1.95 stop_loss           
  TSEM     false_breakdown_w_bottom     $ 163.63 $ 155.45   -5.0% $  -1.98 stop_loss           
  BKNG     false_breakdown_w_bottom     $4302.62 $4087.49   -5.0% $  -1.99 stop_loss           
  CRWD     false_breakdown_w_bottom     $ 413.31 $ 392.64   -5.0% $  -1.99 stop_loss           
======================================================================