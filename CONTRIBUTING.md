# Contributing

Keep changes focused on the submitted paper's experiment protocols. Use a new output directory when changing a frozen configuration, input allocation, or scientific implementation.

Run `python scripts/check_repository.py` and the numerical tests listed in `docs/setup.md`. Changes to aggregation or paper mappings also require `python reproduce.py`.

Preserve paired A/B measurements, reference-bank identities, and nested-size structure. Do not commit downloaded text, token arrays, model weights, generated PDF/TeX files, credentials, or cloud execution exports. The offline evidence in `artifact/` is a versioned snapshot: if modifying it, document the scientific change and update its hash manifest explicitly.

To update publication metadata, edit `paper_metadata.json` and `CITATION.cff`, then update the README citation paragraph. Add only confirmed authors, identifiers, and publication information.
