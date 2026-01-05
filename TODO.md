Harmonization tasks:
TASK 1 - I noticed that for QORC and QLSTM project the runnable function is in runner.py while it is in lib/runtime_entry.py for lib/runner.py for the other projects. Switch to runner.py and check the other projects

log:
Renamed every remaining runtime_entry.py to runner.py (via git mv) in the eight projects that still used the old filename so their history stays intact. While doing so I deleted the temporary wrapper files I had added earlier; the original modules now live directly at runner.py for each project. A repo-wide search shows no lingering imports or references to runtime_entry beyond the note in TODO.md.

TASK 2 - I see that options "--config" and "--outdir" are configured for each projects while they should be global. Move the declaration of these options in runtime_lib/global_cli.json
log:
Added shared definitions for --config and --outdir inside global_cli.json, so the runner now exposes those options once for every project.
Removed the duplicated flag entries from every project cli.json (including the template and all fock_state_expressivity, QORC/QLSTM/QCNN_data_classification, data_reuploading, photonic_QCNN, HQNN_MythOrReality, and qSSL schemas), leaving project files with only project-specific options; q_random_kitchen_sinks now has an empty arguments array since it relied solely on the global flags.
Verified via repo-wide search that no other CLI schema still declares --config or --outdir.

TASK 3 - going further, each project starts logging it starts the training, and then logs in debug the resolved config. Both should be done in the global level for avoiding duplication
log:
Added _log_run_banner inside runtime.py that emits the standardized “Starting <project> run” info message plus the debug dump of the resolved configuration right after CLI overrides are applied and before the project runner executes. This ensures every project gets the same logging without duplicating code.
Removed the now-redundant logger.info("Starting…") / logger.debug("Resolved config…") statements from runner.py, runner.py, runner.py, and runner.py, since the shared runtime now handles those messages centrally.

TASK 4 - check if we still have some code/function/logic in projects - that could be moved globally for consistency and avoiding redundancy
log:
Removed every project-level `setup_seed`/`set_global_seeds` wrapper plus their explicit `train_and_evaluate` calls so seeding now flows exclusively through `runtime_lib.seed.seed_everything`; updated Photonic QCNN and the QLSTM smoke test to rely on the shared helper instead of private shims.
Searched every README.md for lingering references to setup_seed, set_global_seeds, or other project-local seeding helpers. Only the root README.md mentioned seeding metadata.

TASK 5 - I think we can now remove completely the runtime.json - we just expect each module to declare the same function for runner, seed definition and have the same time/run directory convention?
log:
runtime.py now hardcodes the shared layout (configs/defaults.json, configs/cli.json, lib.runner.train_and_evaluate) so individual runtime.json descriptors are unnecessary; seeding always goes through seed_everything, and the runtime raises clear errors when required files are missing. __init__.py exports only run_from_project, and qSSL gained a thin train_and_evaluate adapter so every project exposes the same entry point.
implementation.py discovers runnable projects by checking for configs/defaults.json, configs/cli.json, and runner.py, making --list-projects accurate after the metadata removal.
Tests and helpers (common.py, test_configs.py) now load configs directly and assert that lib.runner.train_and_evaluate exists, matching the new convention.
Deleted every runtime.json file and refreshed documentation (README.md, README.md, README.md, README.md, README.md) plus TODO.md to describe the standardized setup.

TASK 6 - every model running with MerLin / dtype will require conversion of dtype option into torch dtype - we could move these function global, and convert in any case dtype into its torch equivalent, the value of the option could be a pair of the dtype string and the torch converted dtype. This will additional add the possibility of value checking also at the global level, and enable project needing other type of usage of the dtype to still process the string
log:
Added `runtime_lib.dtypes` with alias validation plus helpers (`DtypeSpec`, `dtype_label`, `dtype_torch`, etc.) and taught `runtime_lib/runtime.py` to pass runners a deep-copied config where every `dtype` key becomes a `(label, torch.dtype)` pair; updated the QLSTM runner to consume the shared helper, refreshed the root/template docs to describe the behavior, and recorded the change here.
Added dtypes.py with DtypeSpec, alias validation, and helpers (coerce_dtype_spec, dtype_label, dtype_torch, etc.), and taught runtime.py to deep-copy configs before handing them to runners so every dtype key becomes a validated (label, torch.dtype) pair. The raw JSON config is still what gets logged and snapshot, while the enriched copy flows to project code.
Updated runner.py to drop its local dtype alias map in favor of the shared helper—global and per-model dtype selections now consume DtypeSpec objects and still support the auto fallback for photonic models.
Refreshed developer docs (README.md, README.md) and TODO.md to describe the new behavior and record Task 6 completion; other projects inherit the new dtype normalization automatically.


TASK 7 - make sure all projects have a tests/ directory and that we can test all of them with a single command


