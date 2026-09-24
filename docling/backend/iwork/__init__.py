# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Supporting modules for the Apple iWork backends.

The backends themselves live in :mod:`docling.backend.iwork_backend`, alongside
the other document backends; this package holds what they are built from.

``iwa`` reads the IWA container that Pages, Numbers and Keynote have used since
2013 and ``archives`` the text, table and drawable archives inside it;
``legacy`` reads the ``sf`` vocabulary the same apps wrote before that. Both are
shared, because the apps share their engines. ``content`` models what the
readers produce, and the ``*_iwa`` and ``*_xml`` modules add what one app's own
namespace puts around it.
"""
