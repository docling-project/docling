# Maintainer tools

Tools that give maintainers an overview of the open GitHub issues and PRs, and
act on them with their own `gh` login.

Requirements:

- The [GitHub CLI](https://cli.github.com/), logged in with `gh auth login`.
- Write access to the repository for the actions (comments, CI approval).

Run the tools from the repository root.

## `sync_prs.py`

Rank the open PRs by how close they are to a merge.

```bash
uv run python scripts/tools/sync_prs.py                       # rank only
uv run python scripts/tools/sync_prs.py --comment-conflicts   # comment on conflicts
uv run python scripts/tools/sync_prs.py --approve-ci          # approve safe CI runs
uv run python scripts/tools/sync_prs.py --pr 5115 --pr 5122   # only these PRs
```

A run over 100 PRs takes about 75 seconds.

### Output

The script prints four lists:

1. **Ready to merge**: CI is green, the PR has all the approvals it needs, and
   GitHub reports the merge state `CLEAN`.
2. **CI green, needs approval**: the PR needs at least one more approval.
3. **Merge conflict**: the PR conflicts with its base branch.
4. **Other**: drafts, failing or running CI, CI that waits for a maintainer,
   changes requested, and PRs that GitHub blocks for another reason (for
   example, unresolved review threads).

Each entry shows the reason, for example `1/2 approvals, needs 1 more` or
`CI failed: lint-and-type`.

The script also saves the metadata of each PR to
`.cache/prs/<number>/pr.json`: labels, approvers, checks, merge state, changed
file paths, and the computed rank. It does not download PR code. The folder
`.cache/` is in `.gitignore`.

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
author to rebase on the base branch and shows the git commands. A hidden
marker in the comment records the PR head commit:

- A rerun does not post the same comment again.
- If the author pushes and the conflict stays, the next run posts a new
  comment.

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

### Limits

- The script reads at most 100 changed files per PR. If a larger PR changes
  `tests/data` only after the first 100 files, the script reports 1 required
  approval instead of 2.
- The script does not read review threads. A PR with enough approvals and
  merge state `BLOCKED` goes to "Other". Check it on GitHub.
- GitHub computes mergeability on demand. If GitHub still reports `UNKNOWN`
  after one retry, the PR goes to "Other". Run the script again.
