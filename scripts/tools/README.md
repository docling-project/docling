# Maintainer tools

Tools that give maintainers an overview of the open GitHub issues and PRs, and
act on them with their own `gh` login.

Requirements:

- The [GitHub CLI](https://cli.github.com/), logged in with `gh auth login`.
- Write access to the repository for the actions (comments, CI approval).

Run the tools from the repository root.

| File | Purpose |
| --- | --- |
| `sync_prs.py` | Read the open PRs from GitHub, rank them, find whose turn it is, act on them, and save the data to `.cache/prs/`. |
| `pr_turn.py` | Module: whose turn it is on a PR (maintainers or author). |
| `gh_helpers.py` | Module: `gh` calls, batched GraphQL reads, progress output, and Chrome tabs. |

## `sync_prs.py`

Rank the open PRs by how close they are to a merge.

```bash
uv run python scripts/tools/sync_prs.py                       # rank only
uv run python scripts/tools/sync_prs.py --comment-conflicts   # comment on conflicts
uv run python scripts/tools/sync_prs.py --approve-ci          # approve safe CI runs
uv run python scripts/tools/sync_prs.py --open-ci-greens-in-tabs
uv run python scripts/tools/sync_prs.py --pr 5115 --pr 5122   # only these PRs
uv run python scripts/tools/sync_prs.py --label pdf           # only PRs with this label
uv run python scripts/tools/sync_prs.py --maintainer @me      # your PRs in the turn list
uv run python scripts/tools/sync_prs.py --open-maintainer-turn-in-tabs
```

A run over 100 PRs takes about 90 seconds. Use `--limit 500` to read all
open PRs: about 300 PRs take 4 to 6 minutes. The longer time is when GitHub
must compute the merge state of many PRs, and the script reads them again. Use `-h` to see all options and
their defaults.

While it runs, the script shows status lines (`▸ ...`) and progress bars
with the PR that it reads, on stderr. The report goes to stdout, so
`> report.md` gives a clean Markdown file. Use `--quiet` to turn off the
status lines and progress bars.

### Filter by label

Use `--label` to keep only the PRs with a label. Any label works. These labels
are for formats:

| Label | Format |
| --- | --- |
| `pdf`, `pdf parsing` | PDF (`pdf parsing` is for docling-parse) |
| `docx`, `pptx`, `xlsx` | Microsoft Office |
| `odf` | OpenDocument (ODT, ODS, ODP) |
| `iwork` | Apple Pages, Numbers, and Keynote |
| `html`, `markdown`, `asciidoc`, `latex`, `csv` | Markup and text tables |
| `plain text` | Plain text (the Markdown backend reads it) |
| `epub`, `boxnote`, `calendar`, `email` | EPUB, Box Notes, iCalendar, EML and MSG |
| `image` | Images (PNG, TIFF, JPEG, ...) |
| `audio`, `vtt` | Audio (WAV, MP3) and WebVTT |
| `xml` | All XML formats, also with one of the next labels |
| `jats`, `uspto`, `xbrl`, `doclang` | One XML format |

A format label is present only on PRs that a rule or a maintainer labeled. See
`.github/scripts/pr_topic_labels.py`.

- If you repeat `--label`, a PR needs one of the labels, not all of them.
- GitHub applies the filter, so `--limit` counts only the matching PRs.
- The actions apply only to the matching PRs.
- A label that does not exist gives an empty report, not an error.
- You cannot use `--label` together with `--pr`.
- `-h` lists all labels. This is a fixed list in the script. Update
  `LABELS_HELP` when the repository labels change.

### Output

The script prints four lists, then the PRs where it is the turn of a
maintainer again:

1. **Ready to merge**: CI is green, the PR has all the approvals it needs, and
   GitHub reports the merge state `CLEAN`.
2. **CI green, needs review**: the PR needs at least one more approval.
3. **Merge conflict**: the PR conflicts with its base branch.
4. **Other**: drafts, failing or running CI, CI that waits for a maintainer,
   changes requested, and PRs that GitHub blocks for another reason (for
   example, unresolved review threads).

Lists 2 and 4 have these subsections, by whose turn it is (see
[Maintainer's turn](#maintainers-turn)):

1. **Waits for maintainers to follow up**: a maintainer acted, and the author
   acted after that, or the last maintainer action is an approval. The
   longest wait comes first.
2. **No review started**: no maintainer acted yet.
3. **Author's turn**: the author did not act after the last maintainer action.
4. **Opened by a maintainer or a bot**: the turn is not tracked.

The entries of subsections 1 and 3 also show the last maintainer action. In
list 4, drafts are in the subsections too.

5. **Maintainer's turn again**: a maintainer already acted, and the author
   acted after that, or the last maintainer action is an approval. See [Maintainer's turn](#maintainers-turn). The PRs are
   also in one of the four lists.

Each entry shows the reason, for example `1/2 approvals, needs 1 more` or
`CI failed: lint-and-type`.

The script also saves the metadata of each PR to
`.cache/prs/<number>/pr.json`: labels, approvers, checks, merge state, changed
file paths, whose turn it is (`turn`), and the computed rank. It does not
download PR code. The folder `.cache/` is in `.gitignore`.

### Rules

- **CI is green**: every required status check of the base branch is present,
  and every check is complete with `SUCCESS`, `SKIPPED`, or `NEUTRAL`. The
  `Mergify Merge Protections` check is not part of CI, because it also counts
  approvals.
- **Approvals needed**: the highest value of these rules:
  - the branch ruleset of the base branch (1 for `main`),
  - 2 if the PR changes files in `tests/data` (from
    [`.github/mergify.yml`](../../.github/mergify.yml)),
  - 3 for a breaking change, with `!:` in the title (from
    [`.github/mergify.yml`](../../.github/mergify.yml)).

  Only approvals from owners, members, and collaborators count.

### Actions

All actions are off by default. Without a flag, the script only reads from
GitHub.

#### `--comment-conflicts`

Posts a comment on each PR in the merge-conflict list. The comment asks the
author to rebase on the base branch and shows the git commands.

The script reads the comments of every PR with a merge conflict, also without
this flag. The conflict list shows the state of the notification:

| State | Meaning | With `--comment-conflicts` |
| --- | --- | --- |
| `not notified` | No notification from the script yet. | Posts a comment. |
| `notified 3 h ago, last comment` | Our notification is the last comment. | Skips the PR. |
| `notified 3 d ago, new comments since` | Somebody commented after the notification, but the author did not push. | Skips the PR. |
| `..., author pushed since` | The author pushed after the notification, and the conflict is still there. | Posts a comment (if ours is not the last comment). |

A hidden marker in the comment (`<!-- sync-prs:merge-conflict head=... -->`)
identifies the notification and records the PR head commit. Only PR
conversation comments count, not review comments.

#### `--approve-ci`

GitHub holds the `pull_request` runs of fork PRs with the conclusion
`action_required` until a maintainer approves them. This flag approves these
runs only when all of these conditions are true:

- The PR is not a draft.
- The PR has the label `ai:ci-safe` and not `ai:ci-needs-care`.
- The run is for the current head commit of the PR.
- The label `ai:ci-safe` was added after GitHub created the run. This makes
  sure that the AI triage saw the commit that the run tests.

Without the flag, the script lists the runs that it would approve.

The label `ai:ci-safe` is advisory. See
[`.github/CI_LABELS.md`](../../.github/CI_LABELS.md).

#### `--open-ci-greens-in-tabs`

Opens a new Google Chrome window with a tab for each open PR with green CI.
Your other Chrome windows stay as they are. The order is: ready to merge,
then needs review, then the rest (for example, PRs with a merge conflict or
with changes requested).

The script starts Chrome with the flag `--new-window`:

- On macOS: `open -na "Google Chrome" --args --new-window <urls>`.
- On other systems: the first of `google-chrome`, `google-chrome-stable`,
  `chromium`, `chromium-browser`, or `chrome` on `PATH`.

With 100 PRs this can open more than 50 tabs. Use `--limit` or `--pr` to open
fewer.

### Maintainer's turn

The section "Maintainer's turn again" lists the open PRs where a maintainer
already reviewed or commented, and the author acted after that, or the last
maintainer action is an approval. Drafts are not in this list.

- `--maintainer LOGIN` keeps only the PRs where this maintainer acted. Use
  `@me` for your own `gh` login. It filters only this section, not the four
  lists.
- `--open-maintainer-turn-in-tabs` opens a new Google Chrome window with a tab
  for each PR of this section, the longest wait first (the same way as
  `--open-ci-greens-in-tabs`).

[`pr_turn.py`](pr_turn.py) decides whose turn it is, from the last 100
timeline events and the last 50 review threads of each PR. Each `pr.json` has
one of these states:

| State | Meaning |
| --- | --- |
| `waits for a first review` | No maintainer acted yet. |
| `maintainer's turn` | The author acted after the last maintainer action, or that action is an approval. |
| `author's turn` | The author did not act after the last maintainer action. |
| `not tracked` | An owner, member, collaborator, bot, or `docling-ops` opened the PR. |

- Maintainer actions: reviews, review comments, and conversation comments
  from owners, members, and collaborators. Bot comments do not count.
- Author actions: commits, force pushes, comments, reviews, and review thread
  replies of the PR author.
- Merge commits (for example "Merge branch 'main' into ...") do not count.
- Commits of a maintainer on the author's branch do not count.
- Comments from other contributors do not change whose turn it is.
- When the last maintainer action is an approval, it stays the turn of the
  maintainers (`maintainer's turn`): the PR waits for another approval or for
  the merge. The wait starts at the approval.

The list starts with the PR that waits the longest. The wait starts at the
first author action after the last maintainer action. Each entry shows the
reason from the ranking, the last maintainer action, and what the author did
after it, for example:

```text
- #5402 fix(vlm): Upgrade Built-in Granite Vision Presets to 4.1-4B
  @Yashas-naidu · CI waits for maintainer approval · https://github.com/docling-project/docling/pull/5402
  waits 7 h · last maintainer action: @PeterStaar-IBM, review: changes requested, 21 h ago · since then: 1 comment
```

Limits:

- On a very long PR, the script can miss older actions (see the event limits
  above).
- The script does not read the comment text. An author action can be a plain
  "please review" comment, and a force push can be a rebase without changes.
- The commit date is the date in the commit, not the push date.

### Limits

- The script reads at most 100 changed files per PR. If a larger PR changes
  `tests/data` only after the first 100 files, the script reports 1 required
  approval instead of 2.
- The ranking does not use review threads. A PR with enough approvals and
  merge state `BLOCKED` goes to "Other". Check it on GitHub.
- GitHub computes mergeability on demand. If GitHub still reports `UNKNOWN`
  after one retry, the PR goes to "Other". Run the script again.
