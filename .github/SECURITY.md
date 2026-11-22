# Security and Disclosure Information Policy for the Docling Project

The Docling team and community take security bugs seriously. We appreciate your efforts to responsibly disclose your findings, and will make every effort to acknowledge your contributions.

## Supported Versions

The latest versions of Docling are supported.

### Security

- Participation in the [OpenSSF Best Practices Badge Program](https://bestpractices.coreinfrastructure.org/en/projects/10101) for Free/Libre and FLOSS projects to ensure that we follow current best practices for quality and security
- Use of [HTTPS](https://en.wikipedia.org/wiki/HTTPS) for network communication
- Use of secure protocols for network communication (through the use of HTTPS)
- Up-to-date support for TLS/SSL (through the use of [OpenSSL](https://www.openssl.org/))
- Performance of TLS certificate verification by default before sending HTTP headers with private information (through the use of OpenSSL and HTTPS)
- Distribution of the software via cryptographically signed releases (on the [PyPI](https://pypi.org/), [Quay.io](https://quay.io/organization/docling-project/) and [GHCR.io](https://github.com/orgs/docling-project/packages) package repositories)
- Use of [GitHub](https://github.com/) Issues for vulnerability reporting and tracking

### Analysis

- Use of [Ruff](https://docs.astral.sh/ruff/), [ty](https://docs.astral.sh/ty/) and [Pytest](https://docs.pytest.org/en/7.2.x/) for Python code linting (static and dynamic analysers) on pull requests and builds
- Use of GitHub Issues for bug reporting and tracking

## Scope and Threat Model

This section tells reporters which resource-exhaustion (denial-of-service) reports we treat as vulnerabilities, and how we score them.

### Threat model

Docling is designed to convert untrusted documents in the default configuration.
Operators should still run conversions in isolated workers with memory and CPU-time limits.
These limits are defense in depth. They do not replace the limits inside Docling.

### In scope

We treat a denial-of-service report as a vulnerability when all of these conditions are true:

- A small, attacker-controlled value or structure in the input makes Docling use memory or CPU out of proportion to the input size.
  Examples are decompression bombs, declared spans, repeat counts, point counts or page sizes that Docling uses as allocation sizes, and super-linear algorithms.
- The issue occurs in the default configuration.
- The limits that Docling enables by default do not stop it. These limits are the backend archive limits (`max_file_bytes`, `max_total_bytes`, `max_member_count`) and `max_image_pixels`, at their default values.

Some limits are off by default, and the operator must set them. These limits are `DocumentLimits` (`max_num_pages`, `max_file_size`) and `document_timeout`. They do not make a default-configuration report out of scope.
Docling checks `document_timeout` only between pages or pipeline stages. Thus, a single page or a single native call that does not end is in scope, also when the operator sets `document_timeout`.

### Out of scope

We do not treat these cases as vulnerabilities:

- Cost that grows linearly with the number of bytes the attacker sends. For example, a large ZIP central directory.
- Resource use that stays within the default limits, or within the limits that the operator set.
- Behavior that needs an explicitly enabled option that the documentation marks as risky. We can accept such reports with low severity.
- Vulnerabilities in dependencies that Docling does not make reachable, or whose fix Docling's version pins do not block. Report these to the dependency project.
- Attacks that need code that is already installed in the Python environment. For example, a malicious wheel.

### Scoring

We score document-triggered issues with CVSS 3.1 and `UI:R`, because a user or a service must convert the file.

- Unbounded memory or CPU use in the default configuration is usually `AV:N/AC:L/PR:N/UI:R/S:U/C:N/I:N/A:H`. This gives a score of 6.5 (Medium).
- Bounded cases and cases that need an opt-in option use `A:L`, `AC:H`, or both.
- We do not chain separate third-party vulnerabilities (for example, a Ghostscript sandbox bypass) into the Docling score.

## Reporting a Vulnerability

If you think you've identified a security issue in a Docling project repository, please DO NOT report the issue publicly via the GitHub issue tracker, discussions, or any other public forum.

### Preferred Method: GitHub Private Vulnerability Reporting

**We strongly encourage you to use GitHub's Private Vulnerability Reporting feature**, which provides a secure and streamlined process for disclosing security issues:

1. Navigate to the [Security tab](https://github.com/docling-project/docling/security) of the Docling repository
2. Click on "Report a vulnerability"
3. Fill out the vulnerability report form with as many details as possible
4. Submit the report

This method allows for:
- **Secure communication** directly with the maintainers team
- **Coordinated disclosure** through GitHub's built-in workflow
- **Automatic tracking** of the vulnerability lifecycle
- **Credit attribution** when the vulnerability is published

### Alternative Method: Email Disclosure

Alternatively, you can send an email with as many details as possible to [deepsearch-core@zurich.ibm.com](mailto:deepsearch-core@zurich.ibm.com). This is a private mailing list for the maintainers team.

**Important:** Please do not create a public issue or discuss the vulnerability in any public channel until it has been addressed.

### Security Vulnerability Response

Each report is acknowledged and analyzed by the core maintainers within 3 working days.

Any vulnerability information shared with core maintainers stays within the Docling project and will not be disseminated to other projects unless it is necessary to get the issue fixed.

After the initial reply to your report, the security team will keep you informed of the progress towards a fix and full announcement, and may ask for additional information or guidance.

## Security Alerts

We will send announcements of security vulnerabilities and steps to remediate on the [Docling announcements](https://github.com/docling-project/docling/discussions/categories/announcements).
