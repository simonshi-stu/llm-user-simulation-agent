"""Stage-1 independent simulation export entry for the evaluation contract.

This package runs the accepted P1 reliable workflow offline with a
deterministic fake callable and exports one strict ``SimulationRunV1``
artifact (see :mod:`user_simulator_project.simulation_run_v1`, a mirror of the
canonical contract owned by the evaluation repository). It never calls a real
model, never reads API keys and writes only into new, gitignored local
directories.
"""

STAGE_NAME = "stage1_simulation_export"
