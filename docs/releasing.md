# Releasing SQmdb

SQmdb's `Publish to PyPI` workflow is triggered only when a GitHub Release is
published. Creating or editing a draft does not publish the package. A GitHub
Release is a named announcement for a versioned Git tag; its tag must point to
the reviewed source for the version in `pyproject.toml`.

## One-time publishing setup

Before using the workflow, configure a PyPI Trusted Publisher for the `sqmdb`
project with these exact identity fields:

| Field | Value |
| --- | --- |
| Owner | `benbellman` |
| Repository | `SQmdb` |
| Workflow filename | `publish.yml` |
| GitHub environment | `pypi` |

If `sqmdb` has not been published before, register these values as a pending
publisher in the PyPI account. For an existing PyPI project, add the trusted
publisher in that project's publishing settings.

In the repository's **Settings → Environments**, create an environment named
`pypi` exactly as shown. Under its deployment protection rules, add **Ben
Bellman** as the only required reviewer and save. This approval gate is
configured in GitHub, not in this workflow. With Ben as the sole reviewer,
leave **Prevent self-review** disabled if he needs to approve releases he
initiates; otherwise those deployments will remain blocked. No PyPI API token
or GitHub secret is needed: the publishing job requests a short-lived OIDC
identity and PyPI checks it against the Trusted Publisher registration.

## Publish a stable release

After the release version, source, and CI results are approved:

1. Make sure `project.version` in `pyproject.toml` is the intended release
   version and the release tag matches it, either as `0.1.0` or `v0.1.0`.
2. In GitHub, open **Releases → Draft a new release**, select or create the tag
   for that version, and add release notes.
3. Publish the release. GitHub then runs `.github/workflows/publish.yml`.

The workflow builds the sdist and wheel, transfers them as a short-lived
workflow artifact, and publishes those files to PyPI using the official PyPA
publishing action and Trusted Publishing. Draft releases do not trigger it.
Prereleases run the build job but skip publication; only non-prerelease
releases reach the PyPI publishing job. Do not publish a release until the
repository and PyPI Trusted Publisher setup have been reviewed.
