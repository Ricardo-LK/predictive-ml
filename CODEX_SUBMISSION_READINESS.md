# Codex Task: Make `predictive-ml` Ready for Alignerr Project Submission

## Role and objective

You are working in the repository `Ricardo-LK/predictive-ml`:
https://github.com/Ricardo-LK/predictive-ml

Act as a careful senior software engineer. **Implement** the changes needed to make this existing, independently developed project credible, understandable, reproducible, and demonstrably functional for an application to Alignerr's **Software Engineer – Project Creation** role.

**Submission-ready does not mean production-complete.** This role expects the developer to bring a real, substantial GitHub project and **continue building it** with an AI coding assistant. Preserve meaningful unfinished features and a realistic roadmap. Do not turn it into a generic showcase, rewrite it unnecessarily, or add decorative functionality just to make the repository look bigger.

Work directly in the existing codebase. Begin by checking the current state; the observations below come from a prior review and **may no longer be accurate**.

## Project context

The project is an industrial predictive-maintenance system using the **UCI AI4I 2020 Predictive Maintenance** dataset. It currently includes:

- Python preprocessing, model comparison, cross-validation, threshold selection, training, and MLflow experiment tracking under `src/predictive_ml/`.
- A model inference API under `src/predictive_ml/api.py`, using FastAPI.
- A separate Java/Spring Boot service under `web-service/` that calls the Python inference API.
- Model artifacts under `models/`, a `uv`-based Python project (`pyproject.toml`, `uv.lock`), and a brief `README.md`.

**Preserve** the Python and Java components, the overall problem domain, the owner’s existing code, and the work-in-progress nature of the repository. Do not replace the current stack with a completely different architecture.

## Non-negotiable guardrails

1. **Do not invent results.** Never claim model accuracy, recall, latency, test success, reproducibility, or deployment unless actually measured or verified.
2. **Do not silently change the ML methodology.** Preserve the held-out test set, avoid train/test leakage, and clearly document any statistically significant methodological change.
3. **Do not regenerate or overwrite trained model artifacts without a reason.** If model retraining is necessary, explain why, verify the new model, and record any relevant result changes.
4. **Do not delete the existing `mlflow.db` or model artifacts merely to tidy the repository.** Inspect them for purpose and sensitivity; propose or implement a safe, reversible artifact policy. Do not lose potentially useful experiment history.
5. **Do not introduce paid infrastructure or services.** Use local, free tooling. Do not create external accounts or deploy to a public cloud.
6. **Do not commit secrets, personal data, private datasets, API keys, or corporate materials.** Check tracked files and history-relevant risks at a reasonable level. Avoid exposing project data to additional external services.
7. **Do not imply this project is affiliated with Mitel, PUCPR, Alignerr, or a commercial customer.** If any portion is not owned by the repository owner or is subject to third-party rights, flag it clearly rather than making unsupported ownership claims.
8. **Respect the existing project's development history.** No history rewrites, force-pushes, mass reformatting, or wholesale file replacements. Make cohesive, reviewable changes; commit only if explicitly authorized in the working environment.
9. **Do not add unrelated features or finish every roadmap item.** Keep future work available for genuine AI-assisted development during an Alignerr project.
10. **Never present this project as formally accepted or approved by Alignerr.** The final output is preparation for submission, not a guarantee of acceptance or payment.

## Phase 0: Repository audit (do this first)

- Run `git status`, inspect the recent history, file tree, README, Python dependencies, Java build, and tests. Preserve any existing uncommitted changes; do not overwrite them.
- Map the actual execution paths: how data is loaded, how training runs, how the model artifact is saved/loaded, how the Python API starts, and how the Java service calls it.
- Determine actual Python/Java/runtime requirements and installed dependencies. Prefer the currently selected versions unless there is a **verified** compatibility problem.
- Identify the current state of the project with an explicit distinction between **implemented**, **partially implemented**, and **planned**.
- Create a concise audit summary in the final report, including what is already working, what was broken, and which fixes are necessary.

## Phase 1: Fix correctness and clean-install blockers (highest priority)

Implement the smallest robust fixes needed for a new developer to clone and run the project.

### Python dependency and entry-point correctness

- `src/predictive_ml/api.py` uses `fastapi` and `pydantic`, but these were not explicitly declared in the reviewed `pyproject.toml`. Verify and declare all required direct runtime dependencies, including an ASGI server (e.g. `uvicorn`) if needed.
- Verify `uv.lock` remains synchronized with the dependency manifest. Prefer `uv` for installing, locking, and running Python tasks, consistent with the repo.
- `pyproject.toml` previously declared a `predictive-ml` console script pointing to `predictive_ml:main`, while `__init__.py` only printed a placeholder greeting. Make the CLI meaningful (e.g. documented training command) or remove/replace the misleading entry point. Keep commands straightforward.
- Confirm Python package imports and starting the API work in a clean environment.

### Model training correctness

- Investigate the previously observed reference to `model_cv.cvargs["n_splits"]` in `train.py`. It may be invalid for `RepeatedStratifiedKFold`. Fix the issue only after verifying the actual object/API; obtain logging metadata reliably.
- Check that model selection and threshold selection avoid data leakage and preserve the held-out test set as an unbiased final evaluation.
- Ensure saved model artifacts include all preprocessing steps and any threshold needed for consistent inference.
- Avoid accidentally retraining on the held-out test set or reporting cross-validation scores as test-set results.
- If data downloads need network access, fail with an actionable error and document the requirement; tests should not require network access.

### Python/Java inference contract

- Verify that API input field names, model training feature names, categorical values, and units **actually match**. The UCI AI4I features may have names such as `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]`; the reviewed API used shorter column labels. Do not assume either version: inspect the dataset schema and saved fitted model, then fix the mismatch.
- Make sure `type` values and feature ordering/schema are handled consistently.
- Validate invalid inputs and return useful errors, avoiding unexplained HTTP 500 responses where the input is the cause.
- Check Python `/health`, Python `/predict`, and Java `/api/predictions` for successful responses and reasonable error behavior.
- Confirm `PredictionRequest` and `PredictionResponse` in Java are compatible with the Python API, including number types, property naming, serialization, and validation.
- Preserve the existing Spring Boot service boundary. Improve error handling for unavailable Python inference service and timeouts without unnecessary frameworks.
- Be intentional about model loading. Provide an understandable startup error when the model file is missing or incompatible, instead of silently serving incorrect predictions.

## Phase 2: Add focused, valuable tests

Add tests that prove essential behavior. Prefer a compact high-value test suite over large amounts of boilerplate.

**Python tests:**
- Preprocessing correctly handles expected numerical/categorical inputs and missing values.
- Model/threshold prediction logic behaves consistently, using a **small synthetic fitted model** or a controlled test artifact where practical.
- `/health` responds as expected.
- `/predict` accepts valid requests, emits the expected response schema, and rejects malformed requests cleanly.
- Add at least one regression test guarding against the feature-name/unit mismatch identified above.
- Keep routine tests offline, deterministic, and fast. Do not download UCI data or run full repeated cross-validation in every test.

**Java tests:**
- At least one useful controller/service test beyond the current context-load smoke test.
- Verify request validation, successful forwarding to the Python service (mocked/stubbed), and failure handling when that service cannot be reached.
- Use the project's existing Spring Boot test facilities rather than adding heavy frameworks without need.

Run and record exact test commands and results. If a command cannot run in this environment, say why instead of claiming success.

## Phase 3: Make the repository understandable and reproducible

Rewrite or substantially improve `README.md`, using verified facts only. It should include:

1. **What it does:** a clear 2–4 sentence description of the real industrial failure-prediction problem, intended users, and what the current prototype actually implements.
2. **Architecture:** a compact text diagram or Mermaid diagram showing dataset/training, saved model, FastAPI inference, and Spring Boot client/service.
3. **Stack and data:** Python, scikit-learn, MLflow, FastAPI, Java/Spring Boot, UCI AI4I 2020, plus a link to the original dataset and any important usage/license caveat discovered.
4. **Requirements and setup:** exact supported versions and copy-pasteable commands for Python and Java.
5. **Run it:** instructions to install, train or use the existing model, start FastAPI, start Spring Boot, and test the two endpoints. Clearly describe which steps require internet and which can run offline.
6. **API examples:** at least one **validated** HTTP request/response with correct feature names, types, units, paths, and expected JSON structure; do not fabricate probability values. Use placeholders when necessary.
7. **Evaluation:** briefly describe baseline/models, cross-validation, chosen threshold, and which metrics are calculated; include numeric results **only if verified by a real run**.
8. **Testing:** commands for Python and Java test suites.
9. **Current status / limitations:** distinguish working functionality from planned improvements.
10. **Roadmap:** meaningful tasks to continue later, such as containerization, stronger integration testing, monitoring/drift detection, model versioning, dashboards, cloud deployment. Do **not** implement all roadmap features now.
11. **Project ownership/attribution:** accurately describe original work and third-party dataset/library dependencies. Do not add a license on the owner's behalf without explicit authorization; you may flag that a licensing choice remains open.

Make the README approachable to a reviewer who has never seen the code. Avoid marketing hype and unsupported production claims.

## Phase 4: Minimal developer quality improvements

- Improve `.gitignore` to cover relevant build outputs, cache files, environment files, and temporary logs **without accidentally hiding or deleting important artifacts already tracked**.
- Add practical formatting/lint commands using the existing `ruff` tooling; fix relevant violations without an unrelated full-codebase reformat.
- Where justified, add a simple GitHub Actions CI workflow that runs Python tests/lint and Java tests using compatible toolchains. Favor a reliable, reasonably fast workflow; if current package versions or data downloads prevent stable CI, document the blocker rather than adding a permanently failing workflow.
- Remove obvious placeholder metadata or inaccurate scripts only when this improves clarity.
- Add a brief, realistic `docs/ROADMAP.md` **only if** the README would otherwise become unwieldy; avoid redundant documentation.
- Do not add a frontend, Kubernetes, enterprise observability stack, or other major new capability for this submission-preparation pass.

## Phase 5: Final verification

Before declaring the project submission-ready:

- [ ] The project installs from a clean checkout using documented commands.
- [ ] Python dependencies and lockfile are consistent.
- [ ] Python health and prediction endpoints run against a valid model artifact.
- [ ] The Java endpoint successfully calls the Python inference endpoint, or any environment-specific blocker is precisely documented.
- [ ] Invalid inputs and upstream failures are handled predictably.
- [ ] Training code has no known immediate runtime blocker in the path reviewed; any full-training run not performed is explicitly disclosed.
- [ ] The test suite covers important behavior and passes for all components that can be executed locally.
- [ ] The README setup, commands, request examples, and declared feature units are accurate.
- [ ] No secrets/private data were added to the repository.
- [ ] Existing artifacts and experimental history were not accidentally destroyed.
- [ ] The remaining roadmap contains credible, substantial engineering tasks for continued AI-assisted development.
- [ ] `git diff` has been reviewed for unrelated or accidental changes.

## Deliverables

1. **Implement the required repository changes**, not merely propose them.
2. Provide a concise final summary with:
   - Files changed and why.
   - Important bugs found and fixed.
   - Tests, lint/build/integration commands actually run and their outcomes.
   - Any blocked checks or remaining problems (be explicit).
   - Remaining good first tasks for continuing work with an AI coding assistant.
3. End with a verdict of **`READY`**, **`READY WITH CAVEATS`**, or **`NOT READY`**, justified by the verification results.

## Priority and stopping rule

Order of importance: **runtime correctness > input/model contract > meaningful tests > reproducible setup > honest documentation > optional CI/polish**.

If time or tool constraints prevent completing everything, finish the highest-impact tasks first and report the remainder. Do not hide incomplete work, add superficial features, or claim tests passed without running them.

**Start by auditing the current checkout and presenting the short list of concrete issues. Then implement fixes and verify them. Do not ask for confirmation for routine safe edits; ask only before destructive actions, external deployment, publishing, licensing decisions, or changing the project's fundamental scope.**
