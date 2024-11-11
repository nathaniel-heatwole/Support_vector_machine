# SUPPORT_VECTOR_MACHINE.PY
# Nathaniel Heatwole, PhD (heatwolen@gmail.com)
# Fits a linear support vector machine (SVM), both from scratch and using the built-in functionality in sklearn
# Also uses sklearn to fit non-linear polynomial SVMs of degree two (quadratic) and degree three (cubic)
# Training data: synthetic and randomly generated

import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from colorama import Fore, Style
from matplotlib.backends.backend_pdf import PdfPages

time0 = time.time()
np.random.seed(555)
ver = ''  # version (empty or integer)

topic = 'Support vector machine'
topic_underscore = topic.replace(' ','_')

#--------------#
#  PARAMETERS  #
#--------------#

total_trials = 100           # pairs of randomly chosen points to loop over
iterations_per_trial = 100   # number of iterative optimization steps to perform on each trial
slope_delta = 0.001          # increment to shift the slope (in the iterative optimization)
sigma = 0.75                 # standard deviation for normal distributions generating the random training data
max_iter_sklearn = -1        # maximum number of iterations for sklearn SVM solver (-1 for unlimited)
spatial_granularity = 0.1    # when evaluating the group regions as solid areas

# total points to randomly generate from each group (training data)
n_pts_group1 = 100
n_pts_group2 = 100

# group centroid coordinates (training data)
x_centroid_group1 = 4
x_centroid_group2 = 6
y_centroid_group1 = 7.5
y_centroid_group2 = 2.5

# sample space bounds (region begins at origin) (test data)
x_max = 10
y_max = 10

decimal_places = 3

#-----------------#
#  TRAINING DATA  #
#-----------------#

total_pts = n_pts_group1 + n_pts_group2

group1 = pd.DataFrame()
group2 = pd.DataFrame()

group1['x'] = np.random.normal(x_centroid_group1, sigma, size=n_pts_group1)
group1['y'] = np.random.normal(y_centroid_group1, sigma, size=n_pts_group1)
group2['x'] = np.random.normal(x_centroid_group2, sigma, size=n_pts_group2)
group2['y'] = np.random.normal(y_centroid_group2, sigma, size=n_pts_group2)

group1['group'] = 1
group2['group'] = 2

training_set = pd.concat([group1, group2], axis=0, ignore_index=True)

features = training_set[['x', 'y']]
target = training_set['group']

#----------------#
#  MARGIN WIDTH  #
#----------------#

# function to compute margin width for a given slope value input
def margin_calc(slope):
    fit = training_set.copy(deep=True)
    
    # fit line with this slope through each point in the training data
    fit['y int'] = fit['y'] - (slope * fit['x'])  # rearranged version of the line equation (y = m * x + b)

    # subset by group
    fit_group1 = fit.loc[fit['group'] == 1]
    fit_group2 = fit.loc[fit['group'] == 2]

    # case where group 1 is spatially located ABOVE group 2
    if min(fit_group1['y int']) > max(fit_group2['y int']):
        y_int_group1 = fit_group1['y int'].min()
        y_int_group2 = fit_group2['y int'].max()
        margin = y_int_group1 - y_int_group2
    
    # case where group 2 is spatially located ABOVE group 1
    else:
        y_int_group1 = fit_group1['y int'].max()
        y_int_group2 = fit_group2['y int'].min()
        margin = y_int_group2 - y_int_group1
    
    return margin, y_int_group1, y_int_group2

#----------------#
#  FROM SCRATCH  #
#----------------#

margin_best = -math.inf  # initalize to be the worst possible value (negative infinity - because GREATER margins are sought)

for t1 in range(total_trials):
    # step 1: randomly choose one point from each group
    point1_index = np.random.choice(group1.index)
    point2_index = np.random.choice(group2.index)
    
    # pull coordinates for those particular points
    point1_x = group1['x'][point1_index]
    point1_y = group1['y'][point1_index]
    point2_x = group2['x'][point2_index]
    point2_y = group2['y'][point2_index]
    
    # fine tune slope value
    if (point2_y - point1_y != 0) and (point2_x - point1_x != 0):  # needed to avoid potential divide-by-zero errors
        # slope value to test (negative inverse of slope of line connecting the two randomly chosen points)
        slope_points = (point2_y - point1_y) / (point2_x - point1_x)
        slope_current = -1 / slope_points  # slope is orthogonal to that connecting the points (therefore, is an efficient initial guess for the SVM slope)
        
        # step 2: optimize current slope (parametric variation)
        for t2 in range(iterations_per_trial):     
            # adjacent slopes
            slope_down = slope_current - slope_delta
            slope_up = slope_current + slope_delta

            # margin widths (central and adjacent)
            margin_current, y_int_grp1_current, y_int_grp2_current = margin_calc(slope_current)
            margin_down, y_int_grp1_down, y_int_grp2_down = margin_calc(slope_down)
            margin_up, y_int_grp1_up, y_int_grp2_up = margin_calc(slope_up)
            
            # compare margins with current and adjacent slopes
            if margin_down > margin_current:
                slope_current = slope_down
            elif margin_up > margin_current:
                slope_current = slope_up
        
        # step 3: compare optimized solution for current set of randomly chosen points to running global best solution
        margin_current, y_int_grp1_current, y_int_grp2_current = margin_calc(slope_current)
        if margin_current > margin_best:
            margin_best, slope_best = margin_current, slope_current
            y_int_grp1_best, y_int_grp2_best = y_int_grp1_current, y_int_grp2_current
            y_int_best = 0.5 * (y_int_grp1_best + y_int_grp2_best)  # boundary is midway between the two support vectors

# boundary vector
boundary_vector = pd.DataFrame()
boundary_vector['x'] = np.arange(0, x_max + spatial_granularity, spatial_granularity)
boundary_vector['y'] = (slope_best * boundary_vector['x']) + y_int_best  # equation of line (y = m * x + b)

# group 1 margin vector
support_vector_group1 = pd.DataFrame()
support_vector_group1['x'] = np.arange(0, x_max + spatial_granularity, spatial_granularity)
support_vector_group1['y'] = (slope_best * support_vector_group1['x']) + y_int_grp1_best  # equation of line (y = m * x + b)

# group 2 margin vector
support_vector_group2 = pd.DataFrame()
support_vector_group2['x'] = np.arange(0, x_max + spatial_granularity, spatial_granularity)
support_vector_group2['y'] = (slope_best * support_vector_group2['x']) + y_int_grp2_best  # equation of line (y = m * x + b)

#-----------#
#  SKLEARN  #
#-----------#

# initialize
svm_alt_preds = pd.DataFrame(columns=['x', 'y', 'pred group linear', 'pred group quad', 'pred group cubic'])
boundary_alt_linear = pd.DataFrame(columns=['x', 'y'])
boundary_alt_quad = pd.DataFrame(columns=['x', 'y'])
boundary_alt_cubic = pd.DataFrame(columns=['x', 'y'])

# fit SVMs (linear, quadratic, cubic)
svm_alt_linear = SVC(kernel='linear', max_iter=max_iter_sklearn)
svm_alt_quad = SVC(kernel='poly', degree=2, max_iter=max_iter_sklearn)
svm_alt_cubic = SVC(kernel='poly', degree=3, max_iter=max_iter_sklearn)
svm_alt_linear.fit(features, target)
svm_alt_quad.fit(features, target)
svm_alt_cubic.fit(features, target)

# SVM linear coefficients
w1 = svm_alt_linear.coef_[0][0]
w2 = svm_alt_linear.coef_[0][1]
# weights (w1, w2) and intercept are of the form: (w1 * x) + (w2 * y) + intercept = 0
slope_alt = -w1 / w2
y_int_alt = -svm_alt_linear.intercept_[0] / w2
margin_sklearn, y_int_g1, y_int_g2 = margin_calc(slope_alt)
del y_int_g1, y_int_g2

# initalize
prior_x = 0
prior_y = 0
prior_target_linear = 0
prior_target_quad = 0
prior_target_cubic = 0

# assign each point in the test data (which consists of the entire sample space) to a group using the SVM predictions
for x_pt in np.arange(0, x_max + spatial_granularity, spatial_granularity):
    for y_pt in np.arange(0, y_max + spatial_granularity, spatial_granularity):
        # test point
        obs = pd.DataFrame(columns=['x', 'y'])
        obs[['x', 'y']] = [[x_pt, y_pt]]
        
        # SVM predictions (linear, quadratic, cubic)
        pred_target_linear = svm_alt_linear.predict(obs)[0]
        pred_target_quad = svm_alt_quad.predict(obs)[0]
        pred_target_cubic = svm_alt_cubic.predict(obs)[0]
        svm_alt_preds.loc[len(svm_alt_preds)] = [x_pt, y_pt, pred_target_linear, pred_target_quad, pred_target_cubic]

        # update group boundaries (if predicted group changed relative to last iteration) (uses midpoint as boundary)
        if y_pt > 0:
            if pred_target_linear != prior_target_linear:
                boundary_alt_linear.loc[len(boundary_alt_linear)] = [0.5 * (x_pt + prior_x), 0.5 * (y_pt + prior_y)]  # linear
            if pred_target_quad != prior_target_quad:
                boundary_alt_quad.loc[len(boundary_alt_quad)] = [0.5 * (x_pt + prior_x), 0.5 * (y_pt + prior_y)]      # quadratic
            if pred_target_cubic != prior_target_cubic:
                boundary_alt_cubic.loc[len(boundary_alt_cubic)] = [0.5 * (x_pt + prior_x), 0.5 * (y_pt + prior_y)]    # cubic
                
        # save current iteration values
        prior_target_linear = pred_target_linear
        prior_target_quad = pred_target_quad
        prior_target_cubic = pred_target_cubic
        prior_x = x_pt
        prior_y = y_pt

# add various boundary parameters to dataframes
boundary_alt_linear['y min'] = 0
boundary_alt_quad['y min'] = 0
boundary_alt_cubic['y min'] = 0
boundary_alt_linear['y max'] = y_max
boundary_alt_quad['y max'] = y_max
boundary_alt_cubic['y max'] = y_max

#-------------------#
#  SUMMARY RESULTS  #
#-------------------#

# check if linear separation was successful
if margin_best <= 0:
    print(Fore.RED + '\033[1m' + '\n' + 'ERROR: GROUPS NOT LINEARLY SEPARABLE' + '\n' + Style.RESET_ALL)

# dataframe
svm_linear_summary = pd.DataFrame()
svm_linear_summary.index = ['slope', 'intercept', 'margin width']
svm_linear_summary['from scratch'] = [slope_best, y_int_best, margin_best]
svm_linear_summary['sklearn'] = [slope_alt, y_int_alt, margin_sklearn]
svm_linear_summary = round(svm_linear_summary, decimal_places)

#---------#
#  PLOTS  #
#---------#

# parameters
title_size = 11
axis_labels_size = 8
axis_ticks_size = 8
point_size = 8
legend_size = 8
line_width = 1.25

# function
def format_plot():
    plt.legend(loc='upper right', fontsize=legend_size, ncols=1, facecolor='white', framealpha=1)
    plt.xlabel('X', fontsize=axis_labels_size)
    plt.ylabel('Y', fontsize=axis_labels_size)
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.xticks(fontsize=axis_ticks_size)
    plt.yticks(fontsize=axis_ticks_size)
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.5, zorder=0)
    plt.show(True)

# training data
fig1 = plt.figure()
plt.scatter(group1['x'], group1['y'], marker='*', s=point_size, label='group 1')
plt.scatter(group2['x'], group2['y'], marker='*', s=point_size, label='group 2')
plt.title(topic + ' - training data', fontsize=title_size, fontweight='bold')
format_plot()

# linear SVM (from scratch)
fig2 = plt.figure()
plt.scatter(group1['x'], group1['y'], marker='*', s=point_size, label='group 1')
plt.scatter(group2['x'], group2['y'], marker='*', s=point_size, label='group 2')
plt.plot(boundary_vector['x'], boundary_vector['y'], color='r', linewidth=line_width, label='boundary')
plt.plot(support_vector_group1['x'], support_vector_group1['y'], linewidth=line_width, linestyle='--', color='k', label='SVM margin')
plt.plot(support_vector_group2['x'], support_vector_group2['y'], linewidth=line_width, linestyle='--', color='k')
plt.title(topic + ' - linear (from scratch)', fontsize=title_size, fontweight='bold')
format_plot()

# linear and non-linear SVMs (sklearn)
types = ['linear', 'quadratic', 'cubic']
dfs = [boundary_alt_linear, boundary_alt_quad, boundary_alt_cubic]
for d in range(len(dfs)):
    df = dfs[d]
    # define plot
    if d == 0:
        fig3, ax = plt.subplots()
    elif d == 1:
        fig4, ax = plt.subplots()
    elif d == 2:
        fig5, ax = plt.subplots()
    # populate plot
    if y_int_grp1_best > y_int_grp2_best:
        ax.fill_between(df['x'], df['y max'], df['y'], alpha=0.4, color='#1f77b4', label='group 1', zorder=5)
        ax.fill_between(df['x'], df['y'], df['y min'], alpha=0.4, color='#ff7f0e', label='group 2', zorder=5)
    else:
        ax.fill_between(df['x'], df['y'], df['y min'], alpha=0.4, color='#1f77b4', label='group 1', zorder=5)
        ax.fill_between(df['x'], df['y max'], df['y'], alpha=0.4, color='#ff7f0e', label='group 2', zorder=5)
    plt.scatter(training_set['x'], training_set['y'], marker='*', s=point_size, color='black', label='training', zorder=10)
    plt.title(topic + ' - ' + types[d] + ' (sklearn)', fontsize=title_size, fontweight='bold')
    format_plot()

#----------#
#  EXPORT  #
#----------#

# functions
def console_print(title, df):
    print(Fore.GREEN + '\033[1m' + '\n' + title + Style.RESET_ALL)
    print(df)
def txt_export(title, df, f):
    print(title, file=f)
    print(df, file=f)

# export summary (console, txt)
with open(topic_underscore + '_summary' + ver + '.txt', 'w') as f:
    title = topic.upper() + ' SUMMARY'
    df = svm_linear_summary
    console_print(title, df)
    txt_export(title, df, f)
del f, title, df

# export plots (pdf)
pdf = PdfPages(topic_underscore + '_plots' + ver + '.pdf')
for f in [fig1, fig2, fig3, fig4, fig5]:
    pdf.savefig(f)
pdf.close()
del pdf, f

###

# runtime
runtime_sec = round(time.time() - time0, 2)
if runtime_sec < 60:
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec')
else:
    runtime_min_sec = str(int(np.floor(runtime_sec / 60))) + ' min ' + str(round(runtime_sec % 60, 2)) + ' sec'
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec (' + runtime_min_sec + ')')
del time0


