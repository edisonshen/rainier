======================================================================
QU100 PORTFOLIO BACKTEST — PATTERN-BASED ENTRY/EXIT
======================================================================

  Period:         2022-05-25 to 2026-03-30
  Start capital:  $100.00
  Final capital:  $269.40
  Max positions:  5 (20% each)
  Entry:          Top 2 by pattern confidence
  Patterns:       false_breakdown, false_breakdown_w_bottom
  Hard stop:      11% max loss
  Price mode:     close price only

PERFORMANCE
----------------------------------------------------------------------
  Total trades:       97
  Win rate:           59.8%
  Avg return/trade:   +6.08%
  Median return:      +4.41%
  Total return:       +169.40%
  Sharpe ratio:       1.38
  Max drawdown:       19.38%

  SPY benchmark:      +70.24%
  Alpha:              +99.17%

EXIT REASONS
----------------------------------------------------------------------
  Stop loss:          37
  Target hit:         55
  Max hold:           0
  Pattern invalid:    0
  End of backtest:    5

TOP 5 TRADES
----------------------------------------------------------------------
  Symbol   Pattern                         Entry     Exit   Return      PnL Reason              
  U        false_breakdown_w_bottom     $  19.08 $  43.82 +129.7% $ +38.70 target_hit          
  AMD      false_breakdown_w_bottom     $ 128.24 $ 203.71  +58.9% $ +20.41 target_hit          
  TGT      false_breakdown_w_bottom     $  89.53 $ 118.78  +32.7% $ +16.40 end_of_backtest     
  SBUX     false_breakdown_w_bottom     $  71.99 $ 110.00  +52.8% $ +16.37 target_hit          
  INTC     false_breakdown_w_bottom     $  24.33 $  42.78  +75.8% $ +14.85 target_hit          

WORST 5 TRADES
----------------------------------------------------------------------
  EH       false_breakdown_w_bottom     $  14.17 $  12.42  -12.4% $  -6.15 stop_loss           
  ACN      false_breakdown_w_bottom     $ 280.96 $ 241.21  -14.1% $  -7.57 stop_loss           
  ZS       false_breakdown_w_bottom     $ 169.39 $ 143.28  -15.4% $  -8.14 stop_loss           
  SQQQ     false_breakdown_w_bottom     $ 237.45 $ 161.63  -31.9% $  -9.93 stop_loss           
  CPRI     false_breakdown_w_bottom     $  35.44 $  20.48  -42.2% $ -14.08 stop_loss           
======================================================================