import time

class Timer:
    """Wrapper around perf_counter()."""
    def __init__(self):
        """Instantiate a new Timer."""
        self._events: list[tuple] = list()

    @property
    def events(self):
        """
        Returns all recorded events.
        """
        return self._events

    def reset(self):
        """Clears all previously recorded events."""
        self._events = list()

    def evt(self, label: str | None = None):
        """
        Registers a new event at the current timestamp.

        Parameters
        ----------
        label : str | None
            Optional label for the event.
            Defaults to 'event {idx}'.
        """
        if label is None:
            label = f"event {len(self._events)}"
        self._events.append((label, time.perf_counter()))

    def report(self, labelLength: int = 10):
        """
        Reports times across all events.

        Parameters
        ----------
        labelLength : int
            Formatting print length for the labels.
        """
        for i in range(1, len(self._events)):
            curr = self._events[i]
            prev = self._events[i-1]
            print(f"{prev[0]:<{labelLength}} -> {curr[0]:<{labelLength}}: {(curr[1]-prev[1])*1000:.3f}ms")
        print(f"Total time: {(self._events[-1][1]-self._events[0][1])*1000:.3f}ms")

    def start(self):
        """Convenience method to add an event with label 'start'."""
        if len(self._events) > 0:
            raise RuntimeError("Timer already started")
        self.evt("start")

    def end(self):
        """Convenience method to add an event with label 'end'."""
        self.evt("end")

if __name__ == "__main__":
    timer = Timer()
    timer.start()
    time.sleep(0.1)
    timer.evt("slept")
    time.sleep(0.1)
    timer.end()
    timer.report()
