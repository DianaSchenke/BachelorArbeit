import matplotlib.pyplot as plt
import math
import numpy as np
from numpy.core.defchararray import upper, lower
from sympy.physics.units.definitions.dimension_definitions import information

#below are various functions that make pretty plots using matplotlib


def get_plot_dims(n):
    nrows  = min(int(math.sqrt(n)),4)
    ncols = math.ceil(n/nrows)
    return ncols, nrows

def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels, fontsize=12)
    ax.set_xlim(0.25, len(labels) + 0.75)

def find_outliers(data):
    min_quartile = math.inf
    max_quartile = -math.inf

    for sample in data:
        sample = sample.tolist()
        quartile_1 = np.quantile(sample, 0.25)
        quartile_3 = np.quantile(sample, 0.75)
        min_quartile = min(min_quartile, quartile_1)
        max_quartile = max(max_quartile, quartile_3)

    IQR = np.abs(max_quartile - min_quartile)
    lower_limit = min_quartile - 2 * IQR
    upper_limit = max_quartile + 2 * IQR

    bottom_outliers = []
    top_outliers = []
    for sample in data:
        sample = sample.tolist()
        bottom_outliers.append([x for x in sample if x < lower_limit])
        top_outliers.append([x for x in sample if x > upper_limit])

    if all(not len(l) for l in bottom_outliers):
        bottom_outliers = None

    if all(not len(l) for l in top_outliers):
        top_outliers = None

    return bottom_outliers, top_outliers, lower_limit, upper_limit




def make_pretty_box_plot(data, title, filename, showfliers=True):
    ncols, nrows = get_plot_dims(len(data))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), dpi=600)
    plt.subplots_adjust(left=.2, right=.93, top=.95, bottom=.1)
    fig.suptitle(title, fontsize=16)
    for row in range(nrows):
        for col in range(ncols):
            if nrows==1 and ncols==1:
                ax = axs
            elif nrows ==1:
                ax = axs[col]
            else:
                ax = axs[row, col]
            ax_data = data[ncols*row+col]
            ax.boxplot(ax_data["data"], notch=True, tick_labels = ax_data["tick_labels"], showfliers=showfliers)
            ax.set_xlabel(ax_data["x_label"], fontsize=16)
            ax.set_ylabel(ax_data["y_label"], fontsize=16)
            ax.set_title(ax_data["title"], fontsize=16)
            ax.tick_params(axis='x', which='major', labelsize=16)
            if "format_percent" in ax_data:
                vals = ax.get_yticks()
                ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.savefig(filename)


def make_pretty_violin_plot(data, title, filename, showfliers=True):
    ncols, nrows = get_plot_dims(len(data))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), dpi=600)
    plt.subplots_adjust(left=.07, right=.93, top=.95, bottom=.05)
    fig.suptitle(title, fontsize=16)
    for row in range(nrows):
        for col in range(ncols):
            if nrows==1 and ncols==1:
                ax = axs
            elif nrows ==1:
                ax = axs[col]
            else:
                ax = axs[row, col]
            ax_data = data[ncols*row+col]
            bottom_outliers, top_outliers, lower_limit, upper_limit = find_outliers(ax_data["data"])
            set_axis_style(ax, ax_data["tick_labels"])
            parts = ax.violinplot(ax_data["data"], showextrema=False, widths=0.9)
            for pc in parts['bodies']:
                pc.set_facecolor('lightgrey')
                pc.set_edgecolor('dimgrey')
            quartile1, medians, quartile3 = np.percentile(np.array(ax_data["data"]), [25, 50, 75], axis=1)
            iqr = np.abs(quartile3-quartile1)
            ax.vlines(range(1, len(ax_data["data"])+1), quartile1, quartile3, color='darkgrey', linestyle='-', lw=4, zorder=10)
            ax.vlines(range(1, len(ax_data["data"])+1), quartile1-1.5*iqr, quartile3+1.5*iqr, color='darkgrey', linestyle='-', lw=2, zorder=10)
            ax.scatter(range(1, len(ax_data["data"])+1), medians, color='dimgrey', marker="x", zorder=20)
            ax.set_ylim([lower_limit, upper_limit])
            ax.set_xlabel(ax_data["x_label"], fontsize=13)
            ax.set_ylabel(ax_data["y_label"], fontsize=13)
            ax.set_title(ax_data["title"], fontsize=16)
            if "format_percent" in ax_data:
                vals = ax.get_yticks()
                ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

    plt.savefig(filename)

def make_pretty_violin_plot_with_outliers(data, title, filename, h_line=None):
    ncols, nrows = get_plot_dims(len(data))
    fig = plt.figure(figsize=(5*ncols, 5*nrows), dpi=600)
    plt.subplots_adjust(left=.17, top=.93, bottom=.1)
    subfigs = fig.subfigures(nrows, ncols, wspace=.0, hspace=.0)
    fig.suptitle(title, fontsize=16)
    for row in range(nrows):
        for col in range(ncols):
            ax_data = data[ncols*row+col]
            bottom_outliers, top_outliers, lower_limit, upper_limit = find_outliers(ax_data["data"])
            if nrows==1 and ncols==1:
                subfig = subfigs
            elif nrows ==1:
                subfig = subfigs[col]
            else:
                subfig = subfigs[row, col]

            subfig.supylabel(ax_data["y_label"], fontsize=13)
            subfig.supxlabel(ax_data["x_label"], fontsize=13)
            subfig.suptitle(ax_data["title"])

            if bottom_outliers is None and top_outliers is not None:
                axs = subfig.subplots(2,1,sharex=True, height_ratios=[1,4])
                ax_bottom = None
                ax_centre = axs[1]
                ax_top = axs[0]
            elif bottom_outliers is not None and top_outliers is None:
                axs = subfig.subplots(2,1,sharex=True, height_ratios=[4,1])
                ax_bottom = axs[1]
                ax_centre = axs[0]
                ax_top = None
            elif bottom_outliers is not None and top_outliers is not None:
                axs = subfig.subplots(3,1,sharex=True, height_ratios=[1,3,1])
                ax_bottom = axs[2]
                ax_centre = axs[1]
                ax_top = axs[0]
            else:
                ax_bottom = None
                ax_centre = subfig.subplots(1,1)
                ax_top = None

            if ax_bottom is not None:
                set_axis_style(ax_bottom, ax_data["tick_labels"])
                min_value = math.inf
                max_value = -math.inf
                for i, d in enumerate(bottom_outliers):
                    if d:
                        ax_bottom.scatter(np.full(len(d), i+1), d, facecolors='none', color='dimgrey')
                        min_value = min(min(d), min_value)
                        max_value = max(max(d), max_value)
                ax_bottom.spines["top"].set_visible(False)
                ax_centre.spines["bottom"].set_visible(False)
                ax_centre.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
                y_ticks = [(min_value+max_value)/2]
                if min_value != max_value:
                    total = max_value - min_value
                    ax_bottom.set_ylim([min_value-(.1*total), max_value+(.1*total)])
                    y_ticks = [min_value, (min_value + max_value) / 2, max_value]
                ax_bottom.set_yticks(y_ticks)

            if ax_top is not None:
                ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
                set_axis_style(ax_top, ax_data["tick_labels"])
                min_value = math.inf
                max_value = -math.inf
                for i, d in enumerate(top_outliers):
                    if d:
                        ax_top.scatter(np.full(len(d), i+1), d, facecolors='none', color='dimgrey')
                        min_value = min(min(d), min_value)
                        max_value = max(max(d), max_value)
                ax_centre.spines["top"].set_visible(False)
                ax_top.spines["bottom"].set_visible(False)
                y_ticks = [(min_value+max_value)/2]
                if min_value != max_value:
                    total = max_value - min_value
                    ax_top.set_ylim([min_value-(.1*total), max_value+(.1*total)])
                    y_ticks = [min_value, (min_value + max_value) / 2, max_value]
                ax_top.set_yticks(y_ticks)


            set_axis_style(ax_centre, ax_data["tick_labels"])
            parts = ax_centre.violinplot(ax_data["data"], showextrema=False, widths=0.9)
            for pc in parts['bodies']:
                pc.set_facecolor('lightgrey')
                pc.set_edgecolor('dimgrey')
            quartile1, medians, quartile3 = np.percentile(np.array(ax_data["data"]), [25, 50, 75], axis=1)
            iqr = np.abs(quartile3-quartile1)
            ax_centre.vlines(range(1, len(ax_data["data"])+1), quartile1, quartile3, color='darkgrey', linestyle='-', lw=4, zorder=10)
            ax_centre.vlines(range(1, len(ax_data["data"])+1), quartile1-1.5*iqr, quartile3+1.5*iqr, color='darkgrey', linestyle='-', lw=2, zorder=10)
            ax_centre.scatter(range(1, len(ax_data["data"])+1), medians, color='dimgrey', marker="x", zorder=20)
            ax_centre.set_ylim([lower_limit, upper_limit])
            if "format_percent" in ax_data:
                vals = ax_centre.get_yticks()
                ax_centre.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
                if ax_top is not None:
                    vals = ax_top.get_yticks()
                    ax_top.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
                if ax_bottom is not None:
                    vals = ax_bottom.get_yticks()
                    ax_bottom.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    if h_line is not None:
        plt.axhline(y=h_line,color='black', linestyle='--', lw=1)
    plt.savefig(filename)

def make_split_violin_plot_with_outliers(data_left, data_right, title, left_name, right_name, filename):
    ncols, nrows = get_plot_dims(len(data_left))
    fig = plt.figure(figsize=(5*ncols, 5*nrows), dpi=600)
    plt.subplots_adjust(left=.17, top=.87, bottom=.1)
    subfigs = fig.subfigures(nrows, ncols, wspace=.0, hspace=.0)
    fig.suptitle(title, fontsize=16)
    for row in range(nrows):
        for col in range(ncols):
            ax_data_left = data_left[ncols*row+col]
            ax_data_right = data_right[ncols*row+col]

            bottom_outliers_left, top_outliers_left, lower_limit_left, upper_limit_left = find_outliers(ax_data_left["data"])
            bottom_outliers_right, top_outliers_right, lower_limit_right, upper_limit_right = find_outliers(ax_data_right["data"])
            if nrows==1 and ncols==1:
                subfig = subfigs
            elif nrows ==1:
                subfig = subfigs[col]
            else:
                subfig = subfigs[row, col]

            subfig.supylabel(ax_data_left["y_label"])
            subfig.supxlabel(ax_data_left["x_label"])
            subfig.suptitle(ax_data_left["title"])

            if bottom_outliers_left is None and top_outliers_left is not None and bottom_outliers_right is None and top_outliers_right is not None:
                axs = subfig.subplots(2,2,sharex=True, height_ratios=[1,4])
                axs_bottom = None
                axs_centre = (axs[1][0],axs[1][1])
                axs_top = (axs[0][0], axs[0][1])
            elif bottom_outliers_left is not None and top_outliers_left is None and bottom_outliers_right is not None and top_outliers_right:
                axs = subfig.subplots(2,2,sharex=True, height_ratios=[4,1])
                axs_bottom = (axs[1][0],axs[1][1])
                axs_centre = axs[0]
                axs_top = None
            elif bottom_outliers_left is not None and top_outliers_left is not None and bottom_outliers_right is not None and top_outliers_right is not None:
                axs = subfig.subplots(3,2,sharex=True, height_ratios=[1,3,1])
                axs_bottom = (axs[2][0],axs[2][1])
                axs_centre = (axs[1][0],axs[1][1])
                axs_top = (axs[0][0], axs[0][1])
            else:
                axs_bottom = None
                axs_centre = subfig.subplots(1,2)
                axs_top = None

            if axs_bottom is not None:
                set_axis_style(axs_bottom[0], ax_data_left["tick_labels"])
                set_axis_style(axs_bottom[1], ax_data_right["tick_labels"])
                axs_centre[0].spines["bottom"].set_visible(False)
                axs_centre[1].spines["bottom"].set_visible(False)
                axs_bottom[0].spines["right"].set_visible(False)
                axs_bottom[1].spines["left"].set_visible(False)


                for ax_centre in axs_centre:
                    ax_centre.spines["bottom"].set_visible(False)
                    ax_centre.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

                min_value = math.inf
                max_value = -math.inf
                for i, d in enumerate(bottom_outliers_left):
                    if d:
                        axs_bottom[0].scatter(np.full(len(d), i+1), d, facecolors='none', color='dimgrey')
                        min_value = min(min(d), min_value)
                        max_value = max(max(d), max_value)
                for i, d in enumerate(bottom_outliers_right):
                    if d:
                        axs_bottom[1].scatter(np.full(len(d), i+1), d, facecolors='none', color='dimgrey')
                        min_value = min(min(d), min_value)
                        max_value = max(max(d), max_value)

                y_ticks = [(min_value+max_value)/2]
                axs_bottom[1].tick_params(axis="y", which="both", left=False, right=True, labelleft=False)
                for ax_bottom in axs_bottom:
                    ax_bottom.spines["top"].set_visible(False)
                    if min_value != max_value:
                        total = max_value - min_value
                        ax_bottom.set_ylim([min_value - (.1 * total), max_value + (.1 * total)])
                        y_ticks = [min_value, (min_value + max_value) / 2, max_value]
                    ax_bottom.set_yticks(y_ticks)

            if axs_top is not None:
                set_axis_style(axs_top[0], ax_data_left["tick_labels"])
                set_axis_style(axs_top[1], ax_data_right["tick_labels"])
                axs_centre[0].spines["top"].set_visible(False)
                axs_centre[1].spines["top"].set_visible(False)
                axs_top[0].spines["right"].set_visible(False)
                axs_top[1].spines["left"].set_visible(False)

                min_value = math.inf
                max_value = -math.inf
                for i, d in enumerate(top_outliers_left):
                    if d:
                        axs_top[0].scatter(np.full(len(d), i+1), d, facecolors='none', color='dimgrey')
                        min_value = min(min(d), min_value)
                        max_value = max(max(d), max_value)

                for i, d in enumerate(top_outliers_right):
                    if d:
                        axs_top[1].scatter(np.full(len(d), i+1), d, facecolors='none', color='dimgrey')
                        min_value = min(min(d), min_value)
                        max_value = max(max(d), max_value)

                y_ticks = [(min_value+max_value)/2]
                axs_top[1].tick_params(axis="y", which="both", left=False, right=True, labelleft=False)
                for ax_top in axs_top:
                    ax_top.spines["bottom"].set_visible(False)
                    ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
                    if min_value != max_value:
                        total = max_value - min_value
                        ax_top.set_ylim([min_value-(.1*total), max_value+(.1*total)])
                        y_ticks = [min_value, (min_value + max_value) / 2, max_value]
                    ax_top.set_yticks(y_ticks)
                axs_top[0].set_title(left_name)
                axs_top[1].set_title(right_name)
            else:
                axs_centre[0].set_title(left_name)
                axs_centre[1].set_title(right_name)

            axs_centre[1].tick_params(axis="y", which="both", left=False, right=True, labelleft=False)
            axs_centre[0].spines["right"].set_visible(False)
            axs_centre[1].spines["left"].set_visible(False)

            set_axis_style(axs_centre[0], ax_data_left["tick_labels"])
            set_axis_style(axs_centre[1], ax_data_right["tick_labels"])

            parts = axs_centre[0].violinplot(ax_data_left["data"], showextrema=False, widths=0.9)
            for pc in parts['bodies']:
                pc.set_facecolor('lightgrey')
                pc.set_edgecolor('dimgrey')

            parts = axs_centre[1].violinplot(ax_data_right["data"], showextrema=False, widths=0.9)
            for pc in parts['bodies']:
                pc.set_facecolor('lightgrey')
                pc.set_edgecolor('dimgrey')

            lower_limit = min(lower_limit_left, lower_limit_right)
            upper_limit = max(upper_limit_left, upper_limit_right)

            quartile1, medians, quartile3 = np.percentile(np.array(ax_data_left["data"]), [25, 50, 75], axis=1)
            iqr = np.abs(quartile3-quartile1)
            axs_centre[0].vlines(range(1, len(ax_data_left["data"])+1), quartile1, quartile3, color='darkgrey', linestyle='-', lw=4, zorder=10)
            axs_centre[0].vlines(range(1, len(ax_data_left["data"])+1), quartile1-1.5*iqr, quartile3+1.5*iqr, color='darkgrey', linestyle='-', lw=2, zorder=10)
            axs_centre[0].scatter(range(1, len(ax_data_left["data"])+1), medians, color='dimgrey', marker="x", zorder=20)
            axs_centre[0].set_ylim([lower_limit, upper_limit])

            quartile1, medians, quartile3 = np.percentile(np.array(ax_data_right["data"]), [25, 50, 75], axis=1)
            iqr = np.abs(quartile3-quartile1)
            axs_centre[1].vlines(range(1, len(ax_data_right["data"])+1), quartile1, quartile3, color='darkgrey', linestyle='-', lw=4, zorder=10)
            axs_centre[1].vlines(range(1, len(ax_data_right["data"])+1), quartile1-1.5*iqr, quartile3+1.5*iqr, color='darkgrey', linestyle='-', lw=2, zorder=10)
            axs_centre[1].scatter(range(1, len(ax_data_right["data"])+1), medians, color='dimgrey', marker="x", zorder=20)
            axs_centre[1].set_ylim([lower_limit, upper_limit])




    plt.savefig(filename)

def make_pretty_labeled_scatter_plot(data, title):
    ncols, nrows = get_plot_dims(len(data))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*nrows, 5*ncols), dpi=600)
    fig.suptitle(title, fontsize=16)
    for row in range(nrows):
        for col in range(ncols):
            if nrows == 1:
                ax = axs[col]
                ax_data = data[col]
            else:
                ax =  axs[row, col]
                ax_data = data[ncols*row+col]
            x = ax_data["data"][0]
            y = ax_data["data"][1]
            labels = ax_data["tick_labels"]
            ax.scatter(x, y)
            for i, label in enumerate(labels):
                ax.annotate(label, (x[i], y[i]))
            ax.set_xlabel(ax_data["x_label"], fontsize=12)
            ax.set_ylabel(ax_data["y_label"], fontsize=12)
            ax.set_title(ax_data["title"], fontsize=12)
    plt.show()

def to_tabularx(df, decimals, col_format, caption=None, title=None, index=False, header=True):
    str = df.to_latex(index=index, header=header, float_format=f"%.{decimals}f", column_format=col_format, caption=title)
    str = str.replace("\\begin{tabular}", "\\begin{tabularx}{\\textwidth}")
    str = str.replace("\\toprule", "\\hline")
    str = str.replace("\\midrule", "\\hline")
    str = str.replace("\\bottomrule", "\\hline")
    if caption is not None:
        str = str.replace("\\end{tabular}", "\\end{tabularx}\n"+caption)
    else:
        str = str.replace("\\end{tabular}", "\\end{tabularx}")
    return str