# Block titles as captions

This document demonstrates how AsciiDoc block titles (lines starting with a dot) are handled for different element types.

## Images

A block title before an image is attached as a caption on the picture item.

Figure 1: System Architecture Diagram

<!-- image -->

## Lists

A block title before a list cannot be attached as a structural caption because list groups have no caption slot. It is emitted as a bold paragraph immediately before the list to preserve reading order.

**Important Prerequisites**

- You must install Node.js.
- You need an active API key.

Another block title, this time for an ordered list:

**Steps to complete setup**

1. Clone the repository.
2. Run the install script.
3. Start the server.

## Code blocks

A block title before a literal block is attached as a caption on the code item, mirroring the image behaviour.

```
{ "status": "active" }
```

Example configuration payload

## Tables

A block title before a table is attached as a caption on the table item.

Supported output formats

| Format | Extension |
| - | - |
| Markdown | .md |
| JSON | .json |