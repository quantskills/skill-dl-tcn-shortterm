# Issue tracker: Local Markdown

Issues and specs for this repository live as Markdown files under `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The specification is `.scratch/<feature-slug>/spec.md`
- Implementation issues use one file per ticket:
  `.scratch/<feature-slug>/issues/<NN>-<slug>.md`
- Ticket numbering starts at `01`
- Triage state is recorded in a `Status:` line near the top
- Comments are appended under a `## Comments` heading

## Publishing issues

When a skill says “publish to the issue tracker”, create the appropriate file under `.scratch/<feature-slug>/`, creating directories as needed.

## Fetching issues

Read the explicitly referenced Markdown file. The user will normally provide its path or issue number.

## Wayfinding

- Map: `.scratch/<effort>/map.md`
- Child ticket: `.scratch/<effort>/issues/NN-<slug>.md`
- Ticket type: `Type: research|prototype|grilling|task`
- Execution state: `Status: claimed|resolved`
- Dependencies: `Blocked by: NN, NN`
- Claim a ticket by setting `Status: claimed` before starting
- Resolve it by adding an `## Answer`, setting `Status: resolved`, and recording a context pointer in the map
