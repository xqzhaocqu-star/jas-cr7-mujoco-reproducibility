# Verification evidence

`unit_tests.txt` is the output of the tests run in this standalone export (201 passed).
The fresh run completed 24 s and passed 38/38 independent checks.

`demo_validation.json` records the independent numerical audit of the new 24 s nominal Proposed trial.
`demo_summary.json` records its metrics; `demo_source_manifest.json` fixes the code/model hashes used in that run.
The full `trajectory.npz` stays locally generated and is not committed. Its SHA-256 is in `media/provenance.json`.

The GIF is offline replay of those verified saved states. Two cameras do not represent two trials.
Passing an implementation/physics audit is not a proof of every continuous-time theorem or a claim of zero identification error.
The historical 44-trial metrics under `reference/` are explicitly distinct from this fresh single-trial audit.

`demo_reference_comparison.json` compares the fresh saved state/estimate arrays with the historical nominal run on the same machine; this is not a cross-platform guarantee.
