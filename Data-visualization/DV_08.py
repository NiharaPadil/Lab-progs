#8th a

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import Legend, LegendItem
import numpy as np

output_notebook()

x = np.linspace(0, 10, 100)
y = np.sin(x)
y2 = np.cos(x)

def line_plot_with_annotations():
    p = figure(title="Line Plot with Annotations and Legends", x_axis_label='X', y_axis_label='Y', width=800, height=400)

    line1 = p.line(x, y, line_width=2, color="blue")
    line2 = p.line(x, y2, line_width=2, color="green")

    p.text(x=[5], y=[0.5], text=["Peak of sin(x)"], text_align="center", text_font_size="12pt")
    p.text(x=[8], y=[-0.5], text=["Peak of cos(x)"], text_align="center", text_font_size="12pt")

    legend = Legend(items=[
        LegendItem(label="sin(x)", renderers=[line1]),
        LegendItem(label="cos(x)", renderers=[line2])
    ])

    p.add_layout(legend, 'right')

    show(p)

line_plot_with_annotations()





#8th b


from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import numpy as np

output_notebook()

def scatter_plot():
    x = np.random.rand(50) * 10
    y = np.random.rand(50) * 10
    colors = np.random.choice(['red', 'green', 'blue', 'purple'], size=50)

    p = figure(title="Random Scatter Plot", x_axis_label='X', y_axis_label='Y', width=800, height=400)

    scatter = p.circle(x, y, size=8, color=colors, alpha=0.6)
    scatter.name = "Data Points"

    show(p)

scatter_plot()
