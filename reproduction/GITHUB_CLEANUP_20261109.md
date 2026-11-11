# GitHub publication cleanup — 2026-11-09

Repository: DedMoroz38/BCI; branch: denis.

The initial archive accidentally included a saved Kaggle browser page containing
an embedded Google API key under window.kaggleStackdriverConfig. No key value
is included in this document. The first regex scan omitted Google API keys.

Completed:
- Deleted reproduction/kaggle_runner/notebook-page.html from the archive.
- Added that browser-cache path to the archive's .gitignore and local publisher exclusion.
- Rewrote the single archive commit, preserving the upstream master ancestry.
- Used exact force-with-lease, so concurrent changes would not be overwritten.
- GitHub branch head verified as ae6b06ce60c64a31a46539168bec171737423393.
- The old commit is not contained in any local branch or remote-tracking ref.
- Checked tracked files: zero Google API key-shaped strings.
- One other key-shaped string was accidental image-base64 content in the
  literature review. Wrapped its encoding; decoded image bytes are identical.
- Ran official Gitleaks 8.30.1 after verifying its release checksum.
  It reported 274 generic-api-key findings: 272 bibkey bibliographic identifiers
  in duplicated upstream knowledge-graph files, and two leaderboard metric keys
  in the public competition YAML. All findings belonged to these checked categories.
  No provider-specific secret rule findings remained in the added commit.

Limits:
- GitHub still resolves the old commit by SHA after the rewrite.
- GitHub secret-scanning alerts API returned HTTP 404 for this account; alert
  state could not be read or changed.
- This was an embedded Kaggle frontend telemetry key, not an OAuth credential
  loaded from the local Kaggle login. Ownership/restrictions/revocation were
  not independently verified; no key was tested against any API.
- Revocation was not performed. The code-hosting cleanup does not itself revoke
  a key or remove other people's clones.

## Draft for repository owner to send to GitHub Support

Subject: Request to remove cached commit containing a detected API key

Repository: https://github.com/DedMoroz38/BCI
Affected branch: denis
First changed commit: 8be427017dc459e564581412c7badbdf7bc1a9a9
Replacement commit: ae6b06ce60c64a31a46539168bec171737423393
Affected path: reproduction/kaggle_runner/notebook-page.html, line 106

A saved third-party Kaggle HTML page was accidentally committed. GitHub secret
scanning reported a Google API key embedded in its frontend configuration. The
page has been removed and the single introducing commit was replaced with an
exact force-with-lease update. The affected key value is intentionally omitted
from this request. The old commit remains accessible by its SHA.

Please review whether the cached commit/blob can be purged. We do not control
this third-party key and have not verified its restrictions or revoked it.
No pull request was created by the publication/cleanup process.

Official guidance:
https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository
