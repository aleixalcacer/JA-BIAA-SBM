import numpy as np
import matplotlib as mpl
import seaborn as sns

def create_palette(saturation, value, n_colors, int_colors=3):
        hue = np.linspace(0, 1, n_colors, endpoint=False)
        hue = np.hstack([hue[i::int_colors] for i in range(int_colors)])
        saturation = np.full(n_colors, saturation)
        value = np.full(n_colors, value)
        # convert to RGB
        c = mpl.colors.hsv_to_rgb(np.vstack([hue, saturation, value]).T)
        # Create palette
        palette = mpl.colors.ListedColormap(c)
        return palette


palette = sns.color_palette(create_palette(0.5, 0.95, 4, 1)(np.arange(4)))
models_order = ["BiAA", "SBM", "DBiAA", "DSBM"]
assignments_order = ["hard", "soft"]
