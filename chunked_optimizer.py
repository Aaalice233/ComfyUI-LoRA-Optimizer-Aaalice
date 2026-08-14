"""Private tiled optimizer integration composed from focused mixins."""

try:
    from .chunked_analysis import ChunkedAnalysisMixin
    from .chunked_sources import ChunkedSourceMixin
    from .chunked_strategies import ChunkedStrategyMixin
except ImportError:
    from chunked_analysis import ChunkedAnalysisMixin
    from chunked_sources import ChunkedSourceMixin
    from chunked_strategies import ChunkedStrategyMixin


class ChunkedOptimizerMixin(
        ChunkedSourceMixin,
        ChunkedAnalysisMixin,
        ChunkedStrategyMixin):
    pass
