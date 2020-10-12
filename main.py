import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import pandas as pd

# Generate universe variables
#   * RSI and MACD is has a range [0, 1]
#   * MOMENTUM also has a range one [0,1]
x_rsi = np.arange(0, 11, 1)
x_macd = np.arange(-10, 11, 1)
x_momentum = np.arange(0, 11, 1)

# Generate fuzzy membership functions
rsi_low = fuzz.trapmf(x_rsi, [0, 0, 1, 5])
rsi_middle = fuzz.trimf(x_rsi, [3, 5, 7])
rsi_high = fuzz.trapmf(x_rsi, [5, 9, 10, 10])

macd_above = fuzz.trapmf(x_macd, [-2, 2, 10, 10])
macd_below = fuzz.trapmf(x_macd, [-10, -10, -2, 2])

momentum_bearish = fuzz.trimf(x_momentum, [0, 0, 4])
momentum_neutral = fuzz.trapmf(x_momentum, [2, 4, 6, 8])
momentum_bullish = fuzz.trimf(x_momentum, [6, 10, 10])

# Visualize these universes and membership functions
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_rsi, rsi_low, 'b', linewidth=1.5, label='Low')
ax0.plot(x_rsi, rsi_middle, 'g', linewidth=1.5, label='Middle')
ax0.plot(x_rsi, rsi_high, 'r', linewidth=1.5, label='High')
ax0.set_title('RSI')
ax0.legend()

ax1.plot(x_macd, macd_below, 'b', linewidth=1.5, label='Below')
ax1.plot(x_macd, macd_above, 'g', linewidth=1.5, label='Above')
ax1.set_title('MACD')
ax1.legend()

ax2.plot(x_momentum, momentum_bearish, 'b', linewidth=1.5, label='Bearish')
ax2.plot(x_momentum, momentum_neutral, 'g', linewidth=1.5, label='Neutral')
ax2.plot(x_momentum, momentum_bullish, 'r', linewidth=1.5, label='Bullish')
ax2.set_title('Stock Action')
ax2.legend()

# Turn off top/right axes
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.savefig('plots/membership_functions.png')

results = []

RSI = np.arange(0, 2, 0.5)
MACD = np.arange(0, 2)

for rsi in RSI:
    for macd in MACD:

        print('RSI (' + str(rsi) + ') \t MACD (' + str(macd) + ')')

        # We need the activation of our fuzzy membership functions at these values.
        # This is what fuzz.interp_membership exists for!
        rsi_level_lo = fuzz.interp_membership(x_rsi, rsi_low, rsi)
        rsi_level_md = fuzz.interp_membership(x_rsi, rsi_middle, rsi)
        rsi_level_hi = fuzz.interp_membership(x_rsi, rsi_high, rsi)

        macd_level_below = fuzz.interp_membership(x_macd, macd_below, macd)
        macd_level_above = fuzz.interp_membership(x_macd, macd_above, macd)

        active_rule1 = np.fmin(rsi_level_hi, macd_level_above)
        active_rule2 = np.fmin(macd_level_above, np.fmax(rsi_level_lo, rsi_level_md))

        bullish_activation = np.fmin(active_rule1, momentum_bullish)
        neutral_activation = np.fmin(active_rule2, momentum_neutral)
        bearish_activation = np.fmin(macd_level_below, momentum_bearish)

        momentum0 = np.zeros_like(x_momentum)


        # Visualize this
        fig, ax0 = plt.subplots(figsize=(8, 3))

        ax0.fill_between(x_momentum, momentum0, bullish_activation, facecolor='b', alpha=0.7)
        ax0.plot(x_momentum, momentum_bullish, 'b', linewidth=0.5, linestyle='--', )
        ax0.fill_between(x_momentum, momentum0, neutral_activation, facecolor='g', alpha=0.7)
        ax0.plot(x_momentum, momentum_neutral, 'g', linewidth=0.5, linestyle='--')
        ax0.fill_between(x_momentum, momentum0, bearish_activation, facecolor='r', alpha=0.7)
        ax0.plot(x_momentum, momentum_bearish, 'r', linewidth=0.5, linestyle='--')
        ax0.set_title('Output membership activity')

        # Turn off top/right axes
        for ax in (ax0,):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        plt.tight_layout()

        exp_name = 'RSI (' + str(rsi) + ')_MACD (' + str(macd) + ')'
        plt.savefig('plots/outputs/'+exp_name+'.png')

        # Aggregate all three output membership functions together
        aggregated = np.fmax(bullish_activation,
                             np.fmax(bearish_activation, neutral_activation))

        # Calculate defuzzified result
        momentum_centroid = fuzz.defuzz(x_momentum, aggregated, 'centroid')
        momentum_bisector = fuzz.defuzz(x_momentum, aggregated, 'bisector')
        momentum_activation_centroid = fuzz.interp_membership(x_momentum, aggregated, momentum_centroid)  # for plot
        momentum_activation_bisector = fuzz.interp_membership(x_momentum, aggregated, momentum_bisector)  # for plot

        # Visualize this
        fig, ax0 = plt.subplots(figsize=(8, 3))

        ax0.plot(x_momentum, momentum_bullish, 'b', linewidth=0.5, linestyle='--', )
        ax0.plot(x_momentum, momentum_neutral, 'g', linewidth=0.5, linestyle='--')
        ax0.plot(x_momentum, momentum_bearish, 'r', linewidth=0.5, linestyle='--')

        ax0.fill_between(x_momentum, momentum0, aggregated, facecolor='Orange', alpha=0.7)
        ax0.plot([momentum_centroid, momentum_centroid], [0, momentum_activation_centroid], 'k', color='blue', linewidth=1.5, alpha=0.9, label='centroid')
        ax0.plot([momentum_bisector, momentum_bisector], [0, momentum_activation_bisector], 'k', color='red', linewidth=1.5, alpha=0.9, label='bisector')
        ax0.set_title('Aggregated membership and result (line)')
        ax0.legend()

        # Turn off top/right axes
        for ax in (ax0,):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        plt.tight_layout()
        exp_name = 'RSI (' + str(rsi) + ')_MACD (' + str(macd) + ')'
        plt.savefig('plots/defuzz/'+exp_name+'.png')

        results.append([rsi, macd, momentum_centroid, momentum_bisector])


a = pd.DataFrame(np.asarray(results), columns=['RSI', 'MACD', 'Centroid', 'Bisector'])
a.to_csv("results.csv")







