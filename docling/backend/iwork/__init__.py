# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Support for Apple iWork documents.

``iwa`` reads the IWA container that Pages, Numbers and Keynote have used since
2013; ``pages_backend`` turns a Pages document of either container generation
into a ``DoclingDocument``.
"""

from docling.backend.iwork.pages_backend import IWorkPagesDocumentBackend

__all__ = ["IWorkPagesDocumentBackend"]
