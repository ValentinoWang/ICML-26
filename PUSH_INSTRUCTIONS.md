# Safe Push Instructions

This anonymous export is intended to be pushed from a brand-new git repository with a single clean commit.

## Required workflow

1. Copy this directory to a new standalone location outside the current project tree.
2. In the new location, run `git init`.
3. Set anonymous git identity before the first commit, for example:
   - `git config user.name "anonymous"`
   - `git config user.email "anonymous"`
4. Add only the sanitized export files.
5. Create one clean initial commit.
6. Connect that new repository to the anonymous GitHub repository used by `4open.science`.
7. Push only that new anonymous repository.

## Do not do this

- Do not push from the main project repository.
- Do not reuse existing commit history.
- Do not include `.git` from the current workspace.
- Do not regenerate PDFs with local metadata unless you strip metadata again before push.

## Suggested command sequence

```bash
cp -a rebuttal_exp /path/to/new/location/rebuttal_exp
cd /path/to/new/location/rebuttal_exp
git init
git config user.name "anonymous"
git config user.email "anonymous"
git add .
git commit -m "Initial anonymous rebuttal artifact bundle"
git remote add origin <anonymous-github-remote>
git push -u origin main
```
