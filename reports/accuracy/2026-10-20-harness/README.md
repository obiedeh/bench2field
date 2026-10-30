# Accuracy harness verification

The committed run is an explicitly synthetic scorer fixture,not detector accuracy or a speed promotion. Named execution device bench-5090; CPU scoring only. Source,input/configuration identities and fixed-IoU protocol are in synthetic-run.json. Ground-truth/prediction fixtures are committed under tests/fixtures/accuracy/.

The harness counts missing frames,false positives,duplicates and negative images; rejects unknown IDs,crowds,invalid/nonfinite boxes/scores,hash mismatch and overwrite; preserves undefined metrics for classes with no positives. Both labels and predictions here are handcrafted software fixtures. No actual detector inference,physical label review,FP16/INT8 comparison or retained-speed claim happened. Existing budgets,run reports and report schema remain unchanged. Real accuracy is blocked until independently reviewed labels and model-bound saved predictions on a common evaluation set exist.

14focused tests and 161full tests in the full-suite log cover the implementation and old run compatibility. Changed-file Ruff and Python3.10-target scoped typing pass. Initial caller used the nonexistent package __main__; correct module entrypoint bench2field.cli succeeds without product changes. All failures retained in the external handoff.
