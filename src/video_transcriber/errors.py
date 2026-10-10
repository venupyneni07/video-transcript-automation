"""Actionable errors that can be displayed without a traceback."""


class TranscriberError(Exception):
    """An expected pipeline failure."""


class DependencyError(TranscriberError):
    pass


class DownloadError(TranscriberError):
    pass


class AudioError(TranscriberError):
    pass


class TranscriptionError(TranscriberError):
    pass
