"""Support for Apple iWork documents.

``iwa`` reads the IWA container that Pages, Numbers and Keynote have used since
2013. ``content`` models what a Pages document holds, ``pages_iwa`` and
``pages_xml`` read the two container generations into that model, and
``pages_backend`` turns the result into a ``DoclingDocument``.
"""

from docling.backend.iwork.pages_backend import IWorkPagesDocumentBackend

__all__ = ["IWorkPagesDocumentBackend"]
