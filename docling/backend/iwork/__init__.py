# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Supporting modules for the Apple iWork backends.

The backends themselves live in :mod:`docling.backend.iwork_backend`, alongside
the other document backends; this package holds what they are built from.

``iwa`` reads the IWA container that Pages, Numbers and Keynote have used since
2013, and ``tables`` the ``TST`` table archives they all embed. ``content``
models what a Pages document holds, and ``pages_iwa`` and ``pages_xml`` read the
two container generations into that model.
"""
