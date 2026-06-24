# Contributing

## Branch Workflow

`mveeg` uses a branch-based release workflow:

1. Create feature and fix branches from `origin/develop`.
2. Open pull requests from feature and fix branches into `develop`.
3. Merge accepted work into `develop`.
4. Before releasing, update the package version on `develop`.
5. Open a pull request from `develop` into `main`.
6. Merging into `main` runs CI, builds the package, and creates a GitHub Release.

Do not commit or push directly to `main` or `develop`. Those branches are
integration and release branches and should only change through pull requests.

## Version Updates

The package version is stored in both `pyproject.toml` and
`src/mveeg/__init__.py`. Update both values together on `develop` before opening
the release pull request into `main`.

GitHub Releases use tags derived from the project version, such as `v0.1.1`.
If a tag or release for the current version already exists, the release workflow
fails and the version must be bumped on `develop`.

## Local Checks

Run these checks before opening a pull request:

```bash
uv run python -m pytest
uv build
```

Pull requests into `develop` and `main` run the same checks in GitHub Actions.
