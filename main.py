########################################################################################################################
# Student ID : 28993373
# Name : Bhanuka M.S.V. Gamage
# Assignment 2 : Fuzzy Inference System for Stock Market Momentum
########################################################################################################################

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import os

# Creating the folders for plots
try:
    os.mkdir('plots')
except OSError:
    pass

try:
    os.mkdir('plots/defuzz')
except OSError:
    pass

try:
    os.mkdir('plots/outputs')
except OSError:
    pass


# Generate universe variables
# RSI has a range [0, 10]
# MACD has a range [0, 10]
# MOMENTUM has a range one [0,10]
x_rsi = np.arange(0, 11, 1)
x_macd = np.arange(-10, 11, 1)
x_momentum = np.arange(0, 11, 1)

# Generate fuzzy membership functions

# RSI membership functions
rsi_low = fuzz.trapmf(x_rsi, [0, 0, 1, 5])
rsi_middle = fuzz.trimf(x_rsi, [3, 5, 7])
rsi_high = fuzz.trapmf(x_rsi, [5, 9, 10, 10])

# MACD membership functions
macd_above = fuzz.trapmf(x_macd, [-2, 2, 10, 10])
macd_below = fuzz.trapmf(x_macd, [-10, -10, -2, 2])

# Momemtum membership functions
momentum_bearish = fuzz.trimf(x_momentum, [0, 0, 4])
momentum_neutral = fuzz.trapmf(x_momentum, [2, 4, 6, 8])
momentum_bullish = fuzz.trimf(x_momentum, [6, 10, 10])

########################################################################################################################
# Visualize the membership functions

# RSI
fig, ax0 = plt.subplots(figsize=(8, 3))
ax0.plot(x_rsi, rsi_low, 'b', linewidth=1.5, label='Low')
ax0.plot(x_rsi, rsi_middle, 'g', linewidth=1.5, label='Middle')
ax0.plot(x_rsi, rsi_high, 'r', linewidth=1.5, label='High')
ax0.set_title('RSI')
ax0.legend()
plt.savefig('plots/rsi.eps')

# MACD
fig, ax1 = plt.subplots(figsize=(8, 3))
ax1.plot(x_macd, macd_below, 'b', linewidth=1.5, label='Below')
ax1.plot(x_macd, macd_above, 'g', linewidth=1.5, label='Above')
ax1.set_title('MACD')
ax1.legend()
plt.savefig('plots/macd.eps')

# Momentum
fig, ax2 = plt.subplots(figsize=(8, 3))
ax2.plot(x_momentum, momentum_bearish, 'b', linewidth=1.5, label='Bearish')
ax2.plot(x_momentum, momentum_neutral, 'g', linewidth=1.5, label='Neutral')
ax2.plot(x_momentum, momentum_bullish, 'r', linewidth=1.5, label='Bullish')
ax2.set_title('Momentum')
ax2.legend()
plt.savefig('plots/output.eps')
########################################################################################################################

# Test on different test cases
results = []
# Test options for RSI and MACD
RSI = [0,3.5,5,6.5,10]
MACD = [-10,-1,0,1,10]

# Looping though all the options
for rsi in RSI:
    for macd in MACD:

        print('RSI (' + str(rsi) + ') \t MACD (' + str(macd) + ')')

        # Activation of the fuzzy membership functions at the test values.

        # RSI activation
        rsi_level_lo = fuzz.interp_membership(x_rsi, rsi_low, rsi)
        rsi_level_md = fuzz.interp_membership(x_rsi, rsi_middle, rsi)
        rsi_level_hi = fuzz.interp_membership(x_rsi, rsi_high, rsi)

        # MACD values
        macd_level_below = fuzz.interp_membership(x_macd, macd_below, macd)
        macd_level_above = fuzz.interp_membership(x_macd, macd_above, macd)

        # Rule 1 - If RSI is HIGH and MACD is ABOVE THEN BULLISH
        active_rule1 = np.fmin(rsi_level_hi, macd_level_above)
        bullish_activation = np.fmin(active_rule1, momentum_bullish)

        # Rule 2 - If RSI is LOW OR MIDDLE and MACD is ABOVE THEN NEUTRAL
        active_rule2 = np.fmin(macd_level_above, np.fmax(rsi_level_lo, rsi_level_md))
        neutral_activation = np.fmin(active_rule2, momentum_neutral)

        # Rule 3 - If MACD is BELOW THEN BEARISH
        active_rule3 = macd_level_below
        bearish_activation = np.fmin(active_rule3, momentum_bearish)

        ################################################################################################################
        # Visualize the the output membership activation
        momentum0 = np.zeros_like(x_momentum)
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

        exp_name = 'RSI(' + str(rsi) + ')_MACD(' + str(macd) + ')'
        plt.savefig('plots/outputs/'+exp_name+'.eps')
        ################################################################################################################

        # Aggregate all three output membership functions
        aggregated = np.fmax(bullish_activation,
                             np.fmax(bearish_activation, neutral_activation))


        # Calculate defuzzified result
        momentum_centroid = fuzz.defuzz(x_momentum, aggregated, 'centroid')
        momentum_bisector = fuzz.defuzz(x_momentum, aggregated, 'bisector')

        # Get the activation
        momentum_activation_centroid = fuzz.interp_membership(x_momentum, aggregated, momentum_centroid)
        momentum_activation_bisector = fuzz.interp_membership(x_momentum, aggregated, momentum_bisector)

        ################################################################################################################
        # Visualize the defuzzification result
        fig, ax0 = plt.subplots(figsize=(8, 3))

        ax0.plot(x_momentum, momentum_bullish, 'b', linewidth=0.5, linestyle='--', )
        ax0.plot(x_momentum, momentum_neutral, 'g', linewidth=0.5, linestyle='--')
        ax0.plot(x_momentum, momentum_bearish, 'r', linewidth=0.5, linestyle='--')

        ax0.fill_between(x_momentum, momentum0, aggregated, facecolor='Orange', alpha=0.7)
        ax0.plot([momentum_centroid, momentum_centroid], [0, momentum_activation_centroid], 'k', color='blue', linewidth=1.5, alpha=0.9, label='Centroid (' + str(round(momentum_centroid,4)) + ')')
        ax0.plot([momentum_bisector, momentum_bisector], [0, momentum_activation_bisector], 'k', color='red', linewidth=1.5, alpha=0.9, label='Bisector (' + str(round(momentum_bisector,4)) + ')')
        ax0.set_title('RSI : ' + str(rsi) + ' , MACD : ' + str(macd))
        ax0.legend()

        # Turn off top/right axes
        for ax in (ax0,):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        plt.tight_layout()
        exp_name = 'RSI(' + str(rsi) + ')_MACD(' + str(macd) + ')'
        plt.savefig('plots/defuzz/'+exp_name+'.eps')
        ################################################################################################################

        # Append the results
        results.append([rsi, macd, momentum_centroid, momentum_bisector])

# Save the results as csv
a = pd.DataFrame(np.asarray(results), columns=['RSI', 'MACD', 'Centroid', 'Bisector'])
a.to_csv("results.csv")







