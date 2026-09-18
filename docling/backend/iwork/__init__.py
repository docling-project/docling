# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Supporting modules for the Apple iWork backends.

The backends themselves live in :mod:`docling.backend.iwork_backend`, alongside
the other document backends; this package holds what they are built from.

``iwa`` reads the IWA container that Pages, Numbers and Keynote have used since
2013, and ``tables`` the ``TST`` table archives they all embed. ``content`` and
``numbers_content`` model what a Pages document and a Numbers spreadsheet hold,
and the ``pages_`` and ``numbers_`` readers read the two container generations
into those models.
"""
