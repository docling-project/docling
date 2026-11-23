# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""The process-wide lock for MLX work.

MLX models are not thread-safe. VLM and ASR backends share this lock, so that they
cannot run MLX operations at the same time. This module must not import MLX: it is
on the import path of installs that do not have it.
"""

import threading

MLX_GLOBAL_LOCK = threading.Lock()
