You can run Docling in the cloud without installation using the [Docling Actor][apify] on Apify platform. Simply provide a document URL and get the processed result:

<a href="https://apify.com/vancura/docling?fpr=docling"><img src="https://apify.com/ext/run-on-apify.png" alt="Run Docling Actor on Apify" width="176" height="39" /></a>

```bash
apify call vancura/docling -i '{
  "options": {
    "to_formats": ["md", "json", "html", "text", "doctags"]
  },
  "http_sources": [
    {"url": "https://vancura.dev/assets/actor-test/facial-hairstyles-and-filtering-facepiece-respirators.pdf"},
    {"url": "https://arxiv.org/pdf/2408.09869"}
  ]
}'
```

The Actor stores results in:

* Processed document in key-value store (`OUTPUT_RESULT`)
* Processing logs (`DOCLING_LOG`)
* Dataset record with result URL and status

Read more about the [Docling Actor](.actor/README.md), including how to use it via the Apify API and CLI.

- 💻 [GitHub][github]
- 📖 [Docs][docs]
- 📦 [Docling Actor][apify]

[github]: https://github.com/docling-project/docling/tree/main/.actor/
[docs]: https://github.com/docling-project/docling/tree/main/.actor/README.md
[apify]: https://apify.com/vancura/docling?fpr=docling




