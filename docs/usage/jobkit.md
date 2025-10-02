Docling's document conversion can be executed as distributed jobs using [Docling Jobkit](https://github.com/docling-project/docling-jobkit).

This library provides:

- Pipelines for running jobs with Kueflow pipelines, Ray, or locally.
- Connectors to import and export documents via HTTP endpoints, S3, or Google Drive.

## Usage

### CLI

You can run Jobkit locally via the CLI:

```sh
uv run docling-jobkit-local [configuration-file-path]
```

The configuration file defines:

- Docling conversion options (e.g. OCR settings)
- Source location of input documents
- Target location for the converted outputs

Example configuration file:

```yaml
options:               # Example Docling's conversion options
  do_ocr: false         
sources:               # Source location (here Google Drive)
  - kind: google_drive
    path_id: 1X6B3j7GWlHfIPSF9VUkasN-z49yo1sGFA9xv55L2hSE
    token_path: "./dev/google_drive/google_drive_token.json" 
    credentials_path: "./dev/google_drive/google_drive_credentials.json"  
target:                # Target location (here S3)
  kind: s3
  endpoint: localhost:9000
  verify_ssl: false
  bucket: docling-target
  access_key: minioadmin
  secret_key: minioadmin
```

## Connectors

Connectors are used to import documents for processing with Docling and to export results after conversion.

The currently supported connectors are:

- HTTP endpoints
- S3
- Google Drive

### Google Drive

To use Google Drive as a source or target, you need to enable the API and set up credentials.

Step 1: Enable the [Google Drive API](https://console.cloud.google.com/apis/enableflow?apiid=drive.googleapis.com).

- Go to the Google [Cloud Console](https://console.cloud.google.com/).
- Search for “Google Drive API” and enable it.

Step 2: [Create OAuth credentials](https://developers.google.com/workspace/drive/api/quickstart/python#authorize_credentials_for_a_desktop_application). 

- Go to APIs & Services > Credentials.
- Click “+ Create credentials” > OAuth client ID.
- If prompted, configure the OAuth consent screen with "Audience: External".
- Select application type: "Desktop app".
- Create the application
- Download the credentials JSON and rename it to `google_drive_credentials.json`.

Step 3: Add test users.

- Go to OAuth consent screen > Test users.
- Add your email address.

Step 4: Edit configuration file.

- Edit `credentials_path` with your path to `google_drive_credentials.json`.
- Edit `path_id` with your source or target location. It can be obtained from the URL as follows:
    - Folder: `https://drive.google.com/drive/u/0/folders/1yucgL9WGgWZdM1TOuKkeghlPizuzMYb5` > folder id is `1yucgL9WGgWZdM1TOuKkeghlPizuzMYb5`.
    - File: `https://docs.google.com/document/d/1bfaMQ18_i56204VaQDVeAFpqEijJTgvurupdEDiaUQw/edit` > document id is `1bfaMQ18_i56204VaQDVeAFpqEijJTgvurupdEDiaUQw`.

Step 5: Authenticate via CLI.

- Run the CLI with your configuration file.
- A browser window will open for authentication and gerate a token file that will be save on the configured `token_path` and reused for next runs.
