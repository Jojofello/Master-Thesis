import pandas as pd
import matplotlib
import matplotlib.dates as mdates

def rolling_std(x, w):
    """
    ---------------------------------------------------------------------
    FUNCTION: rolling_std
    ---------------------------------------------------------------------
    Sample standard deviation computed on a rolling window of length w.
    The first w-1 entries are computed with the partial windows that are
    actually available (`min_periods=1`), so the output series has the
    same length as the input.

    INPUTS
        x : 1-D array-like of length T
            Underlying price or return series.
        w : int (positive)
            Rolling-window length in observations.

    OUTPUT
        out : numpy.ndarray of length T
              Rolling sample standard deviation of x with window w.
    ---------------------------------------------------------------------
    """
    s = pd.Series(x)
    return s.rolling(window=w, min_periods=1).std().values

def format_panel(ax, ax2, dates, series, roll, ylabel_l, ylabel_r,
                 title_str, cBlue, cRed, cGrey, is_return=False):
    """
    ---------------------------------------------------------------------
    FUNCTION: format_panel
    ---------------------------------------------------------------------
    Render one of the four panels of the 2x2 grid: a level/return curve
    against the left axis (blue) and the rolling standard deviation
    against a twinned right axis (red).

    INPUTS
        ax        : matplotlib.axes.Axes
                    Primary axes that will host the level/return curve.
        ax2       : matplotlib.axes.Axes
                    Twinned right axes (built via `ax.twinx()`) that will
                    host the rolling standard-deviation curve.
        dates     : pandas.Series of datetime64 of length T
                    Time index used on the x-axis.
        series    : 1-D array-like of length T
                    Either a price or a log-return series.
        roll      : 1-D array-like of length T
                    Rolling standard deviation of `series`.
        ylabel_l  : str
                    Label for the left y-axis (e.g. 'Price (USD)' or
                    'Log-Return').
        ylabel_r  : str
                    Label for the right y-axis (e.g.
                    '3-Month Rolling Std. Dev.').
        title_str : str
                    Bold panel title.
        CBlue     : str
                    Colour Blue
        CRed      : str
                    Colour Red
        CGrey     : str
                    Colour Grey
        is_return : bool, default False
                    If True a horizontal zero line is drawn so positive
                    and negative returns can be told apart at a glance.

    OUTPUT
        None.  The function mutates `ax` / `ax2` in place.
    ---------------------------------------------------------------------
    """
    ax.set_facecolor('#FFFFFF')

    # Left y-axis: level / log-return.
    ax.plot(dates, series, color=cBlue, linewidth=0.7, label='Series')
    if is_return:
        ax.axhline(0, color=cGrey, linewidth=0.35, linestyle='--', zorder=0)
    ax.set_ylabel(ylabel_l, fontsize=7, color=cBlue, labelpad=3)
    ax.tick_params(axis='y', colors=cBlue, labelsize=5.8)
    ax.spines['left'].set_color(cBlue)

    # Right y-axis: rolling standard deviation.
    ax2.plot(dates, roll, color=cRed, linewidth=0.95, label='Rolling Std. Dev.')
    ax2.set_ylabel(ylabel_r, fontsize=6, color=cRed, labelpad=3)
    ax2.tick_params(axis='y', colors=cRed, labelsize=5.8)
    ax2.spines['right'].set_color(cRed)

    # Cosmetic clean-up: hide the top spine on both axes.
    for sp in ['top']:
        ax.spines[sp].set_visible(False)
        ax2.spines[sp].set_visible(False)

    ax.set_title(title_str, fontsize=8, fontweight='bold', pad=5)

    # X-axis: yearly major ticks.
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', labelsize=5.8, rotation=0)
    ax.set_xlim(dates.iloc[0], dates.iloc[-1])

    # Subtle horizontal grid only.
    ax.yaxis.grid(True, linewidth=0.25, color='#E0E0E0', zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Combined legend pulling handles from both axes.
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc='upper left', fontsize=5.5, frameon=False,
              handlelength=1.5, handletextpad=0.4)