"""Read unshifted ORCA arm targets and causal native Sharpa desired targets.

Source: the audited convert_orca_lerobot.py raw-command reader. No action
normalization, differencing, output-frame alignment, or file writes occur here.
The main parquet hand action is a feedback proxy; native desired telemetry
supplies the actual 44 hand command coordinates.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ARM_ACTION_DIM = 14
HAND_ACTION_DIM = 44
HAND_JOINTS_PER_SIDE = 22
HAND_SIDES = (("left", 1), ("right", 0))


def resolve_dataset_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "meta" / "info.json").is_file():
        return path
    candidates = sorted(path.glob("*/meta/info.json"))
    if len(candidates) == 1:
        return candidates[0].parent.parent
    if not candidates:
        raise FileNotFoundError(
            f"Could not find meta/info.json in {path} or one directory below it"
        )
    roots = ", ".join(str(candidate.parent.parent) for candidate in candidates)
    raise ValueError(f"More than one LeRobot dataset found under {path}: {roots}")



def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)



def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records



def fixed_size_list_to_numpy(
    table,
    column_name: str,
    expected_dim: int,
    dtype=np.float32,
) -> np.ndarray:
    column = table[column_name].combine_chunks()
    if not hasattr(column, "values"):
        raise ValueError(f"{column_name} is not a fixed-size list column: {column.type}")
    values = column.values.to_numpy(zero_copy_only=False)
    if values.size != len(column) * expected_dim:
        raise ValueError(
            f"{column_name} contains {values.size} scalar values; "
            f"expected {len(column) * expected_dim}"
        )
    return np.asarray(values, dtype=dtype).reshape(len(column), expected_dim)



def parquet_path(dataset_root: Path, info: dict, episode_id: int) -> Path:
    return dataset_root / info["data_path"].format(
        episode_chunk=episode_id // int(info["chunks_size"]),
        episode_index=episode_id,
    )



def read_main_episode(
    path: Path,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing episode parquet: {path}")
    table = pq.read_table(path, columns=[
        "action",
        "observation.state",
        "frame_index",
        "recording.native_capture_clock_ns",
    ])
    frame_index = np.asarray(
        table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    if not np.array_equal(frame_index, np.arange(len(table))):
        raise ValueError(
            f"frame_index must be contiguous and start at zero in {path}"
        )
    action = fixed_size_list_to_numpy(table, "action", action_dim)
    state = fixed_size_list_to_numpy(table, "observation.state", action_dim)
    native_capture_clock = fixed_size_list_to_numpy(
        table, "recording.native_capture_clock_ns", 3, dtype=np.int64
    )
    if not np.isfinite(action).all() or not np.isfinite(state).all():
        raise ValueError(f"Non-finite action/state values in {path}")
    return action, state, frame_index, native_capture_clock



def load_native_sidecars(dataset_root: Path) -> dict[int, Path]:
    conversion_path = dataset_root / "meta" / "conversion.json"
    conversion = load_json(conversion_path)
    sidecars = {}
    for episode in conversion.get("episodes", []):
        episode_id = int(episode["episode_index"])
        sidecar = dataset_root / episode["native_sidecar"]
        if episode_id in sidecars:
            raise ValueError(f"Duplicate native sidecar for episode {episode_id}")
        sidecars[episode_id] = sidecar
    if not sidecars:
        raise ValueError(f"No native sidecars listed in {conversion_path}")
    return sidecars



def load_hand_server_frame_times(
    sidecar: Path,
    source_indices: np.ndarray,
    workstation_monotonic_ns: np.ndarray,
) -> np.ndarray:
    clock_path = sidecar / "raw" / "clock_samples.jsonl"
    if not clock_path.is_file():
        raise FileNotFoundError(f"Missing native clock samples: {clock_path}")
    clocks = {}
    with clock_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if (
                record.get("op") != "SAMPLE"
                or not record.get("ok")
                or record.get("idx") is None
            ):
                continue
            source_index = int(record["idx"])
            if source_index in clocks:
                raise ValueError(
                    f"Duplicate SAMPLE clock for source index {source_index} in {clock_path}"
                )
            clocks[source_index] = (
                int(record["server_receive_monotonic_ns"]),
                int(record["workstation_monotonic_ns"]),
            )

    server_times = np.empty(len(source_indices), dtype=np.int64)
    for row, (source_index, workstation_time) in enumerate(
        zip(source_indices, workstation_monotonic_ns)
    ):
        source_index = int(source_index)
        if source_index not in clocks:
            raise ValueError(
                f"No native SAMPLE clock for source index {source_index} in {clock_path}"
            )
        server_time, recorded_workstation_time = clocks[source_index]
        if recorded_workstation_time != int(workstation_time):
            raise ValueError(
                f"Clock mismatch for source index {source_index} in {clock_path}: "
                f"main parquet has {int(workstation_time)}, native clock has "
                f"{recorded_workstation_time}"
            )
        server_times[row] = server_time
    return server_times



def read_causal_hand_desired_commands(
    sidecar: Path,
    frame_server_times: np.ndarray,
) -> tuple[np.ndarray, dict]:
    telemetry_path = sidecar / "hand_telemetry.parquet"
    if not telemetry_path.is_file():
        raise FileNotFoundError(f"Missing hand telemetry: {telemetry_path}")

    joint_columns = [f"joint_{index}" for index in range(HAND_JOINTS_PER_SIDE)]
    columns = [
        "kind_name",
        "side_name",
        "complete",
        "status",
        "sequence",
        "source_desired_sequence",
        "source_monotonic_ns",
        *joint_columns,
    ]
    table = pq.read_table(telemetry_path, columns=columns)
    kind = np.asarray(table["kind_name"].to_pylist(), dtype=object)
    side = np.asarray(table["side_name"].to_pylist(), dtype=object)
    complete = np.asarray(table["complete"])
    status = np.asarray(table["status"])
    sequence = np.asarray(table["sequence"], dtype=np.int64)
    source_desired_sequence = np.asarray(
        table["source_desired_sequence"], dtype=np.int64
    )
    source_time = np.asarray(table["source_monotonic_ns"], dtype=np.int64)
    joints = np.column_stack([
        np.asarray(table[column], dtype=np.float64) for column in joint_columns
    ])

    if not complete.all() or not np.equal(status, 0).all():
        raise ValueError(f"Incomplete or nonzero-status hand telemetry in {telemetry_path}")
    if not np.isfinite(joints).all():
        raise ValueError(f"Non-finite hand command values in {telemetry_path}")

    selected_sides = []
    recovered_boundary_records = 0
    selected_boundary_records = 0
    command_ages_ns = []
    for side_name, _ in HAND_SIDES:
        desired_mask = (kind == "desired") & (side == side_name)
        applied_mask = (kind == "applied") & (side == side_name)
        if not desired_mask.any() or not applied_mask.any():
            raise ValueError(
                f"Missing desired/applied telemetry for {side_name} hand in {telemetry_path}"
            )

        desired_sequences = sequence[desired_mask]
        desired_times = source_time[desired_mask]
        desired_joints = joints[desired_mask]
        if len(np.unique(desired_sequences)) != len(desired_sequences):
            raise ValueError(
                f"Duplicate desired sequence for {side_name} hand in {telemetry_path}"
            )

        applied_sequences = source_desired_sequence[applied_mask]
        applied_times = source_time[applied_mask]
        applied_joints = joints[applied_mask]
        desired_by_sequence = {
            int(seq): value for seq, value in zip(desired_sequences, desired_joints)
        }

        # Applied telemetry contains the desired sequence and must reproduce the
        # exact desired vector.  Besides validating provenance, it recovers a
        # command that began just before the capture window.
        synthetic_times = []
        synthetic_joints = []
        for applied_sequence in np.unique(applied_sequences):
            applied_rows = np.flatnonzero(applied_sequences == applied_sequence)
            values = applied_joints[applied_rows]
            if not np.equal(values, values[0]).all():
                raise ValueError(
                    f"Applied command changes within sequence {int(applied_sequence)} "
                    f"for {side_name} hand in {telemetry_path}"
                )
            desired_value = desired_by_sequence.get(int(applied_sequence))
            if desired_value is not None:
                if not np.equal(values[0], desired_value).all():
                    raise ValueError(
                        f"Applied command differs from desired sequence "
                        f"{int(applied_sequence)} for {side_name} hand in "
                        f"{telemetry_path}"
                    )
                continue
            first_applied_row = applied_rows[np.argmin(applied_times[applied_rows])]
            synthetic_times.append(applied_times[first_applied_row])
            synthetic_joints.append(applied_joints[first_applied_row])

        if synthetic_times:
            desired_times = np.concatenate([
                desired_times, np.asarray(synthetic_times, dtype=np.int64)
            ])
            desired_joints = np.concatenate([
                desired_joints, np.stack(synthetic_joints)
            ])
            is_boundary_record = np.concatenate([
                np.zeros(len(desired_sequences), dtype=bool),
                np.ones(len(synthetic_times), dtype=bool),
            ])
        else:
            is_boundary_record = np.zeros(len(desired_sequences), dtype=bool)
        recovered_boundary_records += len(synthetic_times)

        order = np.argsort(desired_times, kind="stable")
        desired_times = desired_times[order]
        desired_joints = desired_joints[order]
        is_boundary_record = is_boundary_record[order]
        positions = np.searchsorted(
            desired_times, frame_server_times, side="right"
        ) - 1
        if np.any(positions < 0):
            first_bad = int(np.flatnonzero(positions < 0)[0])
            raise ValueError(
                f"No causal {side_name} hand command for frame {first_bad} in "
                f"{telemetry_path}"
            )
        ages = frame_server_times - desired_times[positions]
        if np.any(ages < 0):
            raise AssertionError("Causal hand command selection produced a future command")
        command_ages_ns.extend(ages.tolist())
        selected_boundary_records += int(is_boundary_record[positions].sum())
        selected_sides.append(desired_joints[positions].astype(np.float32))

    commands = np.concatenate(selected_sides, axis=1)
    if commands.shape != (len(frame_server_times), HAND_ACTION_DIM):
        raise ValueError(
            f"Unexpected hand command shape {commands.shape} in {telemetry_path}"
        )
    diagnostics = {
        "recovered_boundary_records": recovered_boundary_records,
        "selected_boundary_records": selected_boundary_records,
        "max_command_age_ns": max(command_ages_ns),
    }
    return commands, diagnostics



def read_action_and_state(
    dataset_root: Path,
    info: dict,
    native_sidecars: dict[int, Path],
    episode_id: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    path = parquet_path(dataset_root, info, episode_id)
    action, state, _, native_capture_clock = read_main_episode(path, action_dim)
    if action_dim != ARM_ACTION_DIM + HAND_ACTION_DIM:
        raise ValueError(
            f"Expected {ARM_ACTION_DIM + HAND_ACTION_DIM} action dimensions, "
            f"found {action_dim} in {path}"
        )
    if episode_id not in native_sidecars:
        raise ValueError(f"No native sidecar metadata for episode {episode_id}")
    frame_server_times = load_hand_server_frame_times(
        native_sidecars[episode_id],
        native_capture_clock[:, 0],
        native_capture_clock[:, 1],
    )
    hand_command, diagnostics = read_causal_hand_desired_commands(
        native_sidecars[episode_id], frame_server_times
    )
    action = action.copy()
    action[:, ARM_ACTION_DIM:] = hand_command
    return action, state, diagnostics

