import itertools
import threading

from bokeh import models, palettes, plotting
from bokeh.core import validation

from . import zmq_utils


class BokehWebPlotServer:

    def __init__(self, address):
        validation.silence(validation.warnings.MISSING_RENDERERS)
        # only modify from a Bokeh session callback
        self._sources = {}

        # This is important! Save curdoc() to make sure all threads
        # see the same document.
        self._doc = plotting.curdoc()

        # TODO(Hs293Go): Programmatically generate this
        self._p = [
            plotting.figure(x_axis_label="X (m)", y_axis_label="Y (m)"),
            plotting.figure(x_axis_label="Time (s)", y_axis_label="Z (m)"),
        ]
        self._colors = [
            itertools.cycle(palettes.Category10[10]) for _ in range(len(self._p))
        ]

        self._doc.add_root(models.Row(*self._p))

        self._sub = zmq_utils.Subscriber(address)

        self._thread = threading.Thread(target=self.main_loop)
        self._thread.start()

        self._msg = {}

    def newline(self, line_name, fig_num, opts=None):
        source = models.ColumnDataSource(data=dict(x=[], y=[]))
        self._sources[line_name] = source
        if opts is None:
            opts = {}
        self._p[fig_num].line(
            x="x",
            y="y",
            source=source,
            legend_label=line_name,
            color=next(self._colors[fig_num]),
            line_width=2,
            **opts,
        )

    def update(self):
        if not isinstance(self._msg, dict):
            raise RuntimeError("Expected message in dict format")
        for key, value in self._msg.items():
            fig_num = value["fig_num"]
            if key not in self._sources:
                try:
                    self.newline(key, fig_num, value["opts"])
                except KeyError:
                    self.newline(key, fig_num)

            x, y = value["data"]
            self._sources[key].stream({"x": [x], "y": [y]})

            self.relim(fig_num, x, y)

    def relim(self, fig_num, x, y):
        self.relim_x(fig_num, x)
        self.relim_y(fig_num, y)

    def relim_x(self, fig_num, x):
        xlim = self._p[fig_num].x_range
        if x > xlim.end:
            xlim.end = x * 1.2
        if x < xlim.start:
            xlim.start = x * 1.2

    def relim_y(self, fig_num, y):
        ylim = self._p[fig_num].y_range
        if y > ylim.end:
            ylim.end = y * 1.2
        if y < ylim.start:
            ylim.start = y * 1.2

    def main_loop(self):
        while True:
            # do some blocking computation
            self._msg = self._sub.recv_json()

            # but update the document from a callback
            self._doc.add_next_tick_callback(self.update)
