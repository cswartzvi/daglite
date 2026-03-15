# Zensical Viability Assessment

## What I checked

I validated external adoption using GitHub metadata and repository-topic scanning for `topic:zensical`, then confirmed actual usage by checking each candidate repo for a `zensical.toml` file (or explicit Zensical workflow/config references).

## Confirmed example projects using Zensical

- https://github.com/allthingslinux/tux
- https://github.com/mainsail-crew/docs
- https://github.com/geeksforsocialchange/PlaceCal-Handbook
- https://github.com/lanbaoshen/python-package-template
- https://github.com/maroph/maroph.github.io
- https://github.com/purpleclay/zensical-theme
- https://github.com/cssnr/zensical-action-docs
- https://github.com/hazzuk/karo-docs
- https://github.com/anibridge/anibridge-docs
- https://github.com/radiantBear/writeups

## Viability conclusion

**Viable for Daglite**, with one important caveat:

- Zensical has strong project momentum (active development, significant GitHub stars, and visible community uptake).
- It is designed by the Material for MkDocs team and intentionally serves the same documentation use case.
- However, your current docs relied on `mkdocstrings`; in this migration, those pages are converted to stubs and should be rebuilt with a Zensical-native API docs strategy.

## Recommended next milestones

1. Decide the long-term API documentation pipeline.
2. Fill migration-focus stubs with ownership + timeline.
3. Reintroduce advanced docs features incrementally (math, snippets, search tuning).
