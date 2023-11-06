import matplotlib.pyplot as plt

cm = 1/2.54
AXIS_LABEL_TEXT_SIZE = 8
GRID_ALPHA = .1

plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.titleweight'] = 1
plt.rcParams['axes.titlecolor'] = 'black'
plt.rcParams['grid.alpha'] = .1
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.titlelocation'] = 'left'
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['lines.linewidth'] = .35
plt.rcParams['savefig.transparent'] = True
plt.rcParams['savefig.pad_inches'] = 0
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.format'] = 'tiff'
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['legend.markerscale'] = 3
plt.rcParams['lines.markersize'] = .4

MARKER_ALPHA = .5


def _update_layout(
    figure,
    x: str = 'Wavelength (nm)',
    y: str = 'Intensity (-)'
):

    figure.update_layout(
        legend=dict(
            x=.99,
            y=.95,
            yanchor="top",
            xanchor="right"
        ),
        margin=dict(
            t=50,
            b=60,
            l=60,
            r=10
        ),
        xaxis=dict(
            title=x,
            linecolor='rgba(25,25,25,.4)',
            mirror=True,
            linewidth=2,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(77,77,77,.1)'
        ),
        yaxis=dict(
            title=y,
            linecolor='rgba(25,25,25,.4)',
            mirror=True,
            linewidth=2,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(77,77,77,.1)'
        ),
        plot_bgcolor="#FFF"
    )

    return (figure)
