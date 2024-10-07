from math import exp
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
# import minepy
# from minepy import MINE
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

sns.set_palette('deep')
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

# Print package versions for reproducibility
print('Versions of imported libraries:')
print(f"  matplotlib: {mpl.__version__}")
# print(f"  minepy: {minepy.__version__}")
print(f"  numpy: {np.__version__}")
print(f"  pandas: {pd.__version__}")
print(f"  seaborn: {sns.__version__}")

SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE, TITLE_SIZE = 8, 10, 12, 14
plt.rc('font', size=LARGE_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=TITLE_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=LARGE_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=LARGE_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=LARGE_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('legend', title_fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=TITLE_SIZE)  # fontsize of the figure title

DPI = None
RANDOM_SEED = 42
X_CI = 'sd'
CI = 95
N_BOOT = 1000
ORDER = 1
LOG_X = False
THRESH_PERC = 0.1
P_VAL_THRESH = 0.05

FIG_DIR = '/mnt/c/Users/ASUS/Documents/PhD/Workstation/PaperDrafts/causPath/figs'


def calcCorrs(df, feats, splitFeat='Elements', label='DBoxEX'):
    metricDict = {}
    for feat in feats:
        print(f"Feature: {feat}")
        try:
            df[feat]
        except KeyError:
            print('  Feature not found!')
            continue
        featDict = {}
        for category in df[splitFeat].unique():
            catDF = df[df[splitFeat] == category]
            corr = catDF[label].corr(catDF[feat], method='pearson')
            print(f"  {category} Pearson r: {corr:.2f}")
            featDict[category] = corr
        metricDict[feat] = featDict
    return metricDict


def calcMIs(df, feats, splitFeat='Elements', numHypothesis=1, label='DBoxEX'):
    metricDict = {}
    for feat in feats:
        print(f"Feature: {feat}")
        try:
            df[feat]
        except KeyError:
            print('  Feature not found!')
            continue
        featDict = {}
        for category in df[splitFeat].unique():
            catDF = df.loc[:, [feat, label]][df[splitFeat] == category].dropna(axis=0, how='any', subset=feat)
            if len(catDF) == 0:
                print("    Category doesn't have entry!")
                continue
            mi = mutual_info_regression(pd.DataFrame(catDF[feat]), catDF[label], discrete_features=False, random_state=RANDOM_SEED)
            pVal = exp(-len(catDF[feat]) * mi[0])
            significant = 'significant' if pVal < P_VAL_THRESH / numHypothesis else 'not significant'  # Bonferroni correction
            print(f"  {category} MI: {mi[0]:.4f}, p-value: {pVal:.6f}, {significant}")
            featDict[category] = mi
        metricDict[feat] = featDict
    return metricDict


# def calcMICs(df, feats, alpha=0.6, splitFeat='Elements', label='DBoxEX'):
#     metricDict = {}
#     mine = MINE(alpha=alpha, c=15, est='mic_e')
#     for feat in feats:
#         print(f"Feature: {feat}")
#         try:
#             df[feat]
#         except KeyError:
#             print('  Feature not found!')
#             continue
#         featDict = {}
#         for category in df[splitFeat].unique():
#             catDF = df[df[splitFeat] == category]
#             mine.compute_score(catDF[label], catDF[feat])
#             mic = mine.mic()
#             tic = mine.tic(norm=True)
#             print(f"  {category} MIC_e: {mic:.3f}")
#             print(f"  {category} TIC_e: {tic:.3f}")
#             featDict[category] = mic
#         metricDict[feat] = featDict
#     return metricDict


def plotFeatsBCDcorrBar(featsDBoxCorrDF, NPname, numFeats=None, corrThresh=1.0):
    if not numFeats:
        fig, ax = plt.subplots(figsize=(20, 3), dpi=DPI)
        plt.bar(x=featsDBoxCorrDF[1:numFeats].index, height=featsDBoxCorrDF[1:numFeats], color='#D32F2F', edgecolor='k', linewidth=0.8)
        plt.ylabel(r'Correlation coefficient with $D_{B}$')
        plt.xticks(rotation=90);
        numFeats = 'All'
    else:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=DPI)
        plt.barh(y=featsDBoxCorrDF[1:numFeats].index, width=featsDBoxCorrDF[1:numFeats], color='#D32F2F', edgecolor='k', linewidth=0.8)
        plt.xlabel(r'Correlation coefficient with $D_{B}$')
    plt.grid(linestyle='dotted')
    # plt.title(r'Correlation with $D_{B_{EX}}$');
    plt.savefig(f"{FIG_DIR}/{NPname}FeatDBoxEXTop{numFeats}CorrNoC{int(corrThresh*100)}.png", bbox_inches='tight')


def plotFeatsBCDcorr(df, feats, figNameAppend='MNP', splitFeat='Elements', titleName='Elements (MIC)', 
                     sciNotation=[], logScaleY=[], calcMetric=False, showPlot=True):
    if calcMetric:
        mine = MINE(alpha=0.6, c=15, est='mic_approx') 
    for feat in feats:
        print(f"Feature: {feat}")
        try:
            df[feat]
        except KeyError:
            print('  Feature not found!')
            continue
        # g = sns.lmplot(data=df, x=feat, y='DBoxEX', hue=splitFeat, col=None, row=None, 
        #                palette='deep', col_wrap=None, height=3, aspect=1.35, legend=False, legend_out=None, 
        #                hue_order=None, col_order=None, row_order=None, 
        #                x_estimator=None, x_bins=None, x_ci=X_CI, scatter=True,  markers='o', 
        #                fit_reg=True, ci=CI, n_boot=N_BOOT, units=None, seed=RANDOM_SEED, order=ORDER, 
        #                logistic=False, lowess=False, robust=False, logx=LOG_X, 
        #                x_partial=None, y_partial=None, truncate=True, x_jitter=None, y_jitter=None,
        #                scatter_kws={'s': 10, 'alpha': 0.3}, 
        #                line_kws={'lw': 5, 'linestyle': '--'}, 
        #                facet_kws={'sharex': True, 'sharey': True})
        # g.set_axis_labels(feat, r'$D_B$')
        # legLabelDict = g._legend_data
        # for (i, ele) in enumerate(df[splitFeat].unique()):
        #     eleDF = df[df[splitFeat] == ele]
        #     r2 = eleDF['DBoxEX'].corr(eleDF[feat])
        #     legLabelDict[f"{ele} ({r2:.2f})"] = legLabelDict.pop(ele)
        #     if feat in sciNotation:
        #         plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        #     if logScaleX:
        #         plt.xscale('log')
        # g.add_legend(legend_data=legLabelDict, title=titleName, label_order=legLabelDict.keys(), adjust_subtitles=False)
        # sns.move_legend(g, "lower center", bbox_to_anchor=(0.47, 0.95), ncol=3, frameon=False)
        # g.ax.grid(linestyle='dotted')
        # g.ax.set_axisbelow(True)
        # g.fig.set_dpi = DPI
        # g.savefig(f"figures/BCDcorr/corrBCD{feat}{figNameAppend}.png", bbox_inches='tight')

        # if len(df[splitFeat].unique()) > 1:
        #     g = sns.lmplot(data=df, x=feat, y='DBoxEX', hue=None, col=splitFeat, row=None, 
        #                    palette='deep', col_wrap=None, height=3, aspect=1, legend=False, legend_out=None, 
        #                    hue_order=None, col_order=None, row_order=None, 
        #                    x_estimator=None, x_bins=None, x_ci=X_CI, scatter=True,  markers='o', 
        #                    fit_reg=True, ci=CI, n_boot=N_BOOT, units=None, seed=RANDOM_SEED, order=ORDER, 
        #                    logistic=False, lowess=False, robust=False, logx=LOG_X, 
        #                    x_partial=None, y_partial=None, truncate=True, x_jitter=None, y_jitter=None,
        #                    scatter_kws=None, line_kws=None, facet_kws={'sharex': True, 'sharey': True})
        #     g.set_axis_labels(feat, r'$D_B$')
        #     for (i, ax) in enumerate(g.axes.flatten()):
        #         category = df[splitFeat].unique()[i]
        #         catDF = df[df[splitFeat] == category]
        #         r2 = catDF['DBoxEX'].corr(catDF[feat])
        #         ax.text(0.39, 0.9, f"$R^2$={r2:.2f}", transform=ax.transAxes)
        #         if logScaleX:
        #             ax.set_xscale('log')
        #         ax.grid(linestyle='dotted')
        #         ax.set_axisbelow(True)
        #     g.fig.set_dpi = DPI
        #     g.tight_layout()
        #     g.savefig(f"figures/BCDcorr/corrBCDele{feat}{figNameAppend}.png", bbox_inches='tight')

        g = sns.JointGrid(data=df, x='DBoxEX', y=feat, hue=splitFeat,
                          height=5, ratio=5, space=0.2, 
                          palette=None, hue_order=None, hue_norm=None, 
                          dropna=True, xlim=None, ylim=None, marginal_ticks=False)
        g.plot_joint(sns.scatterplot, s=40, alpha=0.6, linewidth=0.4)  # edgecolor='k', 
        g.plot_marginals(sns.kdeplot, common_norm=False, fill=True)
        # g.ax_marg_y.xaxis.axes.set_xlim(0.0, 150.0)  # For situations when a subset dominates even when common_norm=False
        
        ylabel = feat
        if feat == 'T':
            ylabel = feat + r' (K)'
        if 'Vol' in feat:
            ylabel = feat + r' (m$^3$)'
        elif 'R_' in feat or 'D_' in feat:
            ylabel = feat + ' (nm)'
        g.set_axis_labels(r'$D_B$', ylabel)

        handles, labels = g.ax_joint.get_legend_handles_labels()
        g.ax_joint.legend_.remove()
        legLabelDict = {}
        for (i, category) in enumerate(labels):
            legLabelDict[f"{category}"] = handles[i]
            catDF = df[df[splitFeat] == category]
            if calcMetric:
                # metric = catDF[feat].corr(catDF['DBoxEX'], method='pearson')  # Pearson's correlation 
                mine.compute_score(catDF[feat], catDF['DBoxEX'])
                metric = mine.mic()  # Maximal information coefficient
                print(f"  {category} MIC: {metric:.2f}")
                legLabelDict[f"{category} ({metric:.2f})"] = legLabelDict.pop(category)
            if feat in logScaleY:
                plt.yscale('log')
                g.ax_marg_y.xaxis.offsetText.set_visible(False)
                g.ax_marg_y.yaxis.offsetText.set_visible(False)
            elif feat in sciNotation:
                g.ax_joint.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
                g.ax_joint.yaxis.offsetText.set_horizontalalignment('right')
                g.ax_joint.yaxis.offsetText.set_verticalalignment('bottom')
                g.ax_marg_y.xaxis.offsetText.set_visible(False)
                g.ax_marg_y.yaxis.offsetText.set_visible(False)
        g.ax_joint.legend(handles=legLabelDict.values(), labels=legLabelDict.keys(), title=titleName, loc='best', ncol=1, frameon=True)
        g.ax_joint.legend_.remove()
        # ax = plt.gca()
        # ax.legend(handles=legLabelDict.values(), labels=legLabelDict.keys(), title=titleName)
        # sns.move_legend(ax, "upper left", title=titleName, bbox_to_anchor=(-0.2, 1.23), ncol=1, frameon=False)
        # ax.get_legend().remove()  # Comment out to show legend
        
        g.ax_joint.grid(linestyle='dotted')
        g.ax_joint.set_axisbelow(True)
        g.fig.set_dpi = DPI
        g.savefig(f"figures/BCDcorr/{figNameAppend}/corrJointBCD{feat}{figNameAppend}.png", bbox_inches='tight')

        if showPlot:
            plt.show()


def plotBCDhist(df, figNameAppend='MNP', splitFeat='Elements', titleName='Elements', 
                xFeat='DBoxEX', xLabel=r'$D_B$', xLims=None, threshLines=True, legendXpos=0.47):
    hueOrder = ['BNP', 'TNP', 'MNP'] if splitFeat == 'typeNP' else sorted(df[splitFeat].unique())
    g = sns.displot(data=df, x=xFeat, y=None, hue=splitFeat, row=None, col=None,
                    weights=None, kind='kde', rug=False, rug_kws=None, log_scale=None, 
                    legend=True, palette='deep', hue_order=hueOrder, hue_norm=None, color=None, col_wrap=None, 
                    row_order=None, col_order=None, height=3, aspect=1.2, facet_kws=None, common_norm=False)
    if threshLines:
        print(f"{figNameAppend}:\n")
        for (i, category) in enumerate(hueOrder):
            catDF = df[xFeat][df[splitFeat] == category]
            smoothThreshVal = catDF.min() + (catDF.max() - catDF.min()) * THRESH_PERC
            roughThreshVal = catDF.min() + (catDF.max() - catDF.min()) * (1 - THRESH_PERC)
            print(f"\t{category}:\n\t\tSmooth: {smoothThreshVal}\tRough: {roughThreshVal}\n\t\tMin: {catDF.min()}\tMax: {catDF.max()}")
            g.refline(x=smoothThreshVal, color=sns.color_palette()[i])
            g.refline(x=roughThreshVal, color=sns.color_palette()[i])
    g.set_axis_labels(xLabel, 'Probability density')
    g.axes[0][0].grid(linestyle='dotted')
    if xLims is not None:
        g.axes[0][0].set_xlim(xLims[0], xLims[1])
    # ax = plt.legend(title=splitFeat, labels=df[splitFeat].unique(), ncols=3) # loc='lower center', bbox_to_anchor=(0.5, -0.35), frameon=False
    # legLabelDict = {}
    # for (i, ele) in enumerate(df[splitFeat].unique()):
    #     legLabelDict[f"{ele}"] = ax.legend_handles[i]
    # ax.remove()
    # g.add_legend(legend_data=legLabelDict, title=titleName, label_order=legLabelDict.keys(), adjust_subtitles=False)
    # sns.move_legend(g, "lower center", title=titleName, bbox_to_anchor=(0.63, 0.68), ncol=1, frameon=True)  # For rough vs smooth nanoparticle
    sns.move_legend(g, "lower center", title=titleName, bbox_to_anchor=(legendXpos, -0.12), ncol=3, frameon=False)

    g.fig.set_dpi = DPI
    g.savefig(f"figures/BCDhist/histBCD{figNameAppend}.png", bbox_inches='tight')


def compareRoughSmoothStats(df, percDiffThresh=30):
    bcds = df['DBoxEX']
    smoothThreshVal = bcds.min() + (bcds.max() - bcds.min()) * THRESH_PERC
    roughThreshVal = bcds.min() + (bcds.max() - bcds.min()) * (1 - THRESH_PERC)
    roughNPsDF = df[bcds > roughThreshVal]
    smoothNPsDF = df[bcds < smoothThreshVal]
    roughSmoothNPsDF = pd.concat([roughNPsDF.describe().loc['mean', :], smoothNPsDF.describe().loc['mean', :], 
                                  df.describe().loc['max', :], df.describe().loc['min', :]], axis=1)
    roughSmoothNPsDF.columns = ['Rough Mean', 'Smooth Mean', 'Max', 'Min']
    roughSmoothNPsDF['AbsDiff'] = roughSmoothNPsDF.apply(lambda f: abs(f['Rough Mean'] - f['Smooth Mean']), axis=1)
    roughSmoothNPsDF['PercDiff'] = roughSmoothNPsDF.apply(lambda f: f['AbsDiff'] / abs(f['Max'] - f['Min']) * 100, axis=1)
    sigDiffDF = roughSmoothNPsDF[roughSmoothNPsDF['PercDiff'] > percDiffThresh]
    return sigDiffDF

