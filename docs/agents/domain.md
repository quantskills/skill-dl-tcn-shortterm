# Domain Docs

This repository uses a single-context domain-document layout.

## Before exploring

Read:

- Root `CONTEXT.md`
- Relevant ADRs under `docs/adr/`

If `docs/adr/` does not yet exist, proceed silently. ADRs are created lazily when decisions are actually resolved.

## Layout

```text
/
├── CONTEXT.md
├── docs/
│   └── adr/
└── src/
```

`CONTEXT.md` owns current domain terminology and context. `PROGRAM.md` remains the durable project mission and boundary contract.

## Vocabulary

Use domain concepts exactly as defined in `CONTEXT.md`. Avoid introducing synonyms that conflict with its glossary.

If a required concept is missing, either reconsider whether it belongs to the project or record the gap for domain modeling.

## ADR conflicts

If proposed work contradicts an existing ADR, surface the conflict explicitly instead of silently overriding the decision.
