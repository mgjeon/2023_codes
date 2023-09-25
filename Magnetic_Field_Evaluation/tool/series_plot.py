import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter, date2num
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries


def plot_with_flares(series_results, noaanum: int, fig_path, include_C=False):
    x_dates = date2num(series_results["date"])
    date_format = DateFormatter("%m/%d-%H:%M")

    flares = Fido.search(
        a.Time(min(series_results["date"]), max(series_results["date"])),
        a.hek.EventType("FL"),
        a.hek.OBS.Observatory == "GOES",
    )["hek"]

    filtered_results = flares[
        "event_starttime", "event_peaktime", "event_endtime", "fl_goescls", "ar_noaanum"
    ]
    filtered_results_noaa = filtered_results[filtered_results["ar_noaanum"] == noaanum]

    fig, full_axs = plt.subplots(
        5, 2, figsize=(20, 15), gridspec_kw={"width_ratios": [1, 0.05]}
    )
    axs = full_axs[:, 0]
    [ax.axis("off") for ax in full_axs[:, 1]]

    for ax in axs:
        ax.xaxis_date()
        ax.set_xlim(x_dates[0], x_dates[-1])
        ax.xaxis.set_major_formatter(date_format)

    fig.autofmt_xdate()

    msize = 2.5

    ax = axs[0]
    ax.plot(
        x_dates,
        np.array(series_results["total_free_energy"]) * 1e-32,
        "ko-",
        markersize=msize,
    )
    ax.set_ylabel("Free Energy\n[$10^{32}$ erg]")

    ax = axs[1]
    ax.plot(
        x_dates, np.array(series_results["norm_laplacian_B_max_0"]), "ko-", markersize=2
    )
    ax.set_ylabel("max " + r"|$\nabla^2 \mathbf{B}$|" + "\n in the whole domain")

    ax = axs[2]
    # The cropped (1 pix) domain is the domain whose border (1 pixel) is removed
    # to reduce the boundary effect of the finite difference method.
    ax.plot(
        x_dates,
        np.array(series_results["norm_laplacian_B_max_1"]),
        "ko-",
        markersize=msize,
    )
    ax.set_ylabel(
        "max "
        + r"|$\nabla^2 \mathbf{B}$|"
        + " [G/Mm$^2$]"
        + "\n in the cropped (1 pix) domain"
    )

    ax = axs[3]
    # The cropped (3 pix) domain is the domain whose border (3 pixel) is removed
    # to reduce the boundary effect of the finite difference method.
    ax.plot(
        x_dates,
        np.array(series_results["norm_laplacian_B_max_3"]),
        "ko-",
        markersize=msize,
    )
    ax.set_ylabel(
        "max "
        + r"|$\nabla^2 \mathbf{B}$|"
        + " [G/Mm$^2$]"
        + "\n in the cropped (3 pix) domain"
    )

    ax = axs[4]
    # The cropped (5 pix) domain is the domain whose border (5 pixel) is removed
    # to reduce the boundary effect of the finite difference method.
    ax.plot(
        x_dates,
        np.array(series_results["norm_laplacian_B_max_5"]),
        "ko-",
        markersize=msize,
    )
    ax.set_ylabel(
        "max "
        + r"|$\nabla^2 \mathbf{B}$|"
        + " [G/Mm$^2$]"
        + "\n in the cropped (5 pix) domain"
    )

    if include_C is True:
        my_labels = {"X": "X", "M": "M", "C":"C"}
        for st, pt, et, cl in zip(
            filtered_results_noaa["event_starttime"],
            filtered_results_noaa["event_peaktime"],
            filtered_results_noaa["event_endtime"],
            filtered_results_noaa["fl_goescls"],
        ):
            if cl[0] == "X":
                for ax in axs:
                    ax.axvline(
                        x=date2num(pt.datetime),
                        linestyle="dotted",
                        c="red",
                        label=my_labels["X"],
                    )
                    ax.axvspan(
                        date2num(st.datetime),
                        date2num(et.datetime),
                        alpha=0.2,
                        color="red",
                    )
                    my_labels["X"] = "_nolegend_"
            elif cl[0] == "M":
                for ax in axs:
                    ax.axvline(
                        x=date2num(pt.datetime),
                        linestyle="dotted",
                        c="green",
                        label=my_labels["M"],
                    )
                    ax.axvspan(
                        date2num(st.datetime),
                        date2num(et.datetime),
                        alpha=0.2,
                        color="green",
                    )
                    my_labels["M"] = "_nolegend_"
            elif cl[0] == "C":
                for ax in axs:
                    ax.axvline(
                        x=date2num(pt.datetime),
                        linestyle="dotted",
                        c="blue",
                        label=my_labels["C"],
                    )
                    ax.axvspan(
                        date2num(st.datetime),
                        date2num(et.datetime),
                        alpha=0.2,
                        color="blue",
                    )
                    my_labels["C"] = "_nolegend_"
    elif include_C is False:
        my_labels = {"X": "X", "M": "M"}
        for st, pt, et, cl in zip(
            filtered_results_noaa["event_starttime"],
            filtered_results_noaa["event_peaktime"],
            filtered_results_noaa["event_endtime"],
            filtered_results_noaa["fl_goescls"],
        ):
            if cl[0] == "X":
                for ax in axs:
                    ax.axvline(
                        x=date2num(pt.datetime),
                        linestyle="dotted",
                        c="red",
                        label=my_labels["X"],
                    )
                    ax.axvspan(
                        date2num(st.datetime),
                        date2num(et.datetime),
                        alpha=0.2,
                        color="red",
                    )
                    my_labels["X"] = "_nolegend_"
            elif cl[0] == "M":
                for ax in axs:
                    ax.axvline(
                        x=date2num(pt.datetime),
                        linestyle="dotted",
                        c="green",
                        label=my_labels["M"],
                    )
                    ax.axvspan(
                        date2num(st.datetime),
                        date2num(et.datetime),
                        alpha=0.2,
                        color="green",
                    )
                    my_labels["M"] = "_nolegend_"

    fig.suptitle(f"NOAA {noaanum}")
    fig.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
