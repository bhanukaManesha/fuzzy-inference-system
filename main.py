import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Generate universe variables
#   * RSI and MACD is has a range [0, 1]
#   * STOCK ACTION also has a range one [0,1]
x_rsi = np.arange(0, 11, 1)
x_macd = np.arange(0, 11, 1)
x_stockaction = np.arange(0, 11, 1)


# Generate fuzzy membership functions
rsi_low = fuzz.trapmf(x_rsi, [0, 0, 1, 4])
rsi_middle = fuzz.trimf(x_rsi, [2, 5, 8])
rsi_high = fuzz.trapmf(x_rsi, [6, 9, 10, 10])

macd_high = fuzz.trapmf(x_macd, [2, 7, 10, 10])
macd_low = fuzz.trapmf(x_macd, [0, 0, 3, 8])

stockaction_buy = fuzz.trimf(x_stockaction, [0, 0, 4])
stockaction_hold = fuzz.trapmf(x_stockaction, [0, 3, 7, 10])
stockaction_sell = fuzz.trimf(x_stockaction, [6, 10, 10])

# Visualize these universes and membership functions
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_rsi, rsi_low, 'b', linewidth=1.5, label='Low')
ax0.plot(x_rsi, rsi_middle, 'g', linewidth=1.5, label='Middle')
ax0.plot(x_rsi, rsi_high, 'r', linewidth=1.5, label='High')
ax0.set_title('RSI')
ax0.legend()

ax1.plot(x_macd, macd_low, 'b', linewidth=1.5, label='Low')
ax1.plot(x_macd, macd_high, 'g', linewidth=1.5, label='High')
ax1.set_title('MACD')
ax1.legend()

ax2.plot(x_stockaction, stockaction_buy, 'b', linewidth=1.5, label='Buy')
ax2.plot(x_stockaction, stockaction_hold, 'g', linewidth=1.5, label='Hold')
ax2.plot(x_stockaction, stockaction_sell, 'r', linewidth=1.5, label='Sell')
ax2.set_title('Stock Action')
ax2.legend()

# Turn off top/right axes
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

RSI = 5
MACD = 5


# We need the activation of our fuzzy membership functions at these values.
# This is what fuzz.interp_membership exists for!
rsi_level_lo = fuzz.interp_membership(x_rsi, rsi_low, RSI)
rsi_level_md = fuzz.interp_membership(x_rsi, rsi_middle, RSI)
rsi_level_hi = fuzz.interp_membership(x_rsi, rsi_high, RSI)

macd_level_lo = fuzz.interp_membership(x_macd, macd_low, MACD)
macd_level_hi = fuzz.interp_membership(x_macd, macd_high, MACD)

active_rule2 = np.fmin(macd_level_lo, np.fmax(rsi_level_lo,rsi_level_md))
active_rule3 = np.fmin(macd_level_hi, np.fmax(rsi_level_lo,rsi_level_md))

sell_activation = np.fmin(rsi_level_hi, stockaction_sell)
hold_activation = np.fmin(active_rule2, stockaction_hold)
buy_activation = np.fmin(active_rule3, stockaction_buy)

stockaction0 = np.zeros_like(x_stockaction)


# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(x_stockaction, stockaction0, sell_activation, facecolor='b', alpha=0.7)
ax0.plot(x_stockaction, stockaction_sell, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_stockaction, stockaction0, hold_activation, facecolor='g', alpha=0.7)
ax0.plot(x_stockaction, stockaction_hold, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(x_stockaction, stockaction0, buy_activation, facecolor='r', alpha=0.7)
ax0.plot(x_stockaction, stockaction_buy, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Output membership activity')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

# Aggregate all three output membership functions together
aggregated = np.fmax(sell_activation,
                     np.fmax(buy_activation, hold_activation))

# Calculate defuzzified result
stockaction = fuzz.defuzz(x_stockaction, aggregated, 'centroid')
tip_activation = fuzz.interp_membership(x_stockaction, aggregated, stockaction)  # for plot

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(x_stockaction, stockaction_sell, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(x_stockaction, stockaction_hold, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_stockaction, stockaction_buy, 'r', linewidth=0.5, linestyle='--')

ax0.fill_between(x_stockaction, stockaction0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([stockaction, stockaction], [0, tip_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result (line)')

print(tip_activation)

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

plt.show()







