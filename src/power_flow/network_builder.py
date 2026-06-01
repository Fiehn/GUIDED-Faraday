from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pandapower as pp

_DEFAULT_TOPOLOGY = Path("data/topology/cleaned/lv_topology.csv")
_DEFAULT_MAPPING = Path("data/topology/cleaned/Cleaned_SM_Topology_mapping.json")

# Nominal voltage for LV network (kV)
_VN_KV = 0.4


def build_feeder_network(
    feeder_id: str,
    topology_path: Path = _DEFAULT_TOPOLOGY,
    mapping_path: Path = _DEFAULT_MAPPING,
) -> tuple[pp.pandapowerNet, dict[str, int]]:
    """Build a pandapower network for *feeder_id* and return (net, meter_to_load_index).

    Raises ValueError if feeder_id is absent from the topology CSV.
    Warns and skips meters whose cabinet node is not in the feeder graph.
    """
    topology_path = Path(topology_path)
    mapping_path = Path(mapping_path)

    df = pd.read_csv(topology_path)
    feeder_label = f"LvFeeder.{feeder_id}"
    feeder_rows = df[df["lv_feeder"] == feeder_label]

    if feeder_rows.empty:
        raise ValueError(
            f"Feeder ID '{feeder_id}' ('{feeder_label}') not found in {topology_path}"
        )

    with open(mapping_path) as fh:
        mapping: dict[str, dict] = json.load(fh)

    net = pp.create_empty_network(f_hz=50.0, sn_mva=1.0)

    # --- Slack bus (HV side of the transformer — ext_grid here so power flows
    #     through the transformer and loading_percent is correctly populated) ---
    substation_id = feeder_rows["secondary_substation"].iloc[0]
    transformer_bus = pp.create_bus(net, vn_kv=20.0, name=f"HV_{substation_id}", type="b")
    pp.create_ext_grid(net, bus=transformer_bus, vm_pu=1.0, name="slack")

    # LV slack bus (transformer secondary / feeder root)
    slack_bus = pp.create_bus(net, vn_kv=_VN_KV, name=substation_id, type="b")

    # Transformer: use highest capacity row (all rows share same transformer in a feeder)
    transformer_kva = feeder_rows["transformer_capacity"].iloc[0]
    pp.create_transformer_from_parameters(
        net,
        hv_bus=transformer_bus,
        lv_bus=slack_bus,
        sn_mva=transformer_kva / 1000.0,
        vn_hv_kv=20.0,
        vn_lv_kv=_VN_KV,
        vkr_percent=1.0,
        vk_percent=4.0,
        pfe_kw=0.0,
        i0_percent=0.0,
        name=f"Transformer_{feeder_id}",
    )

    # --- Identify the feeder root edge (node1 == 'LvFeeder.<id>') ---
    # The LvFeeder node IS the slack bus; its cable row gives the root link parameters.
    root_mask = feeder_rows["node1"] == feeder_label
    cable_rows = feeder_rows[~root_mask]  # ordinary cabinet-to-cabinet edges

    # --- Cabinet buses (exclude the LvFeeder node itself — that's the slack) ---
    all_nodes: set[str] = set(cable_rows["node1"]) | set(cable_rows["node2"])
    cabinet_bus: dict[str, int] = {}
    for node in all_nodes:
        bus_idx = pp.create_bus(net, vn_kv=_VN_KV, name=node, type="b")
        cabinet_bus[node] = bus_idx

    # --- Lines (one per cabinet-to-cabinet cable row) ---
    for _, row in cable_rows.iterrows():
        from_bus = cabinet_bus[row["node1"]]
        to_bus = cabinet_bus[row["node2"]]
        length_km = row["cable_length"] / 1000.0
        pp.create_line_from_parameters(
            net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=length_km,
            r_ohm_per_km=row["resistance"],
            x_ohm_per_km=row["reactance"],
            c_nf_per_km=0.0,
            max_i_ka=row["cable_capacity"] / 1000.0,
            name=row.get("cable_id", ""),
        )

    # --- Root link: slack bus → first cabinet, using real cable parameters if available ---
    if root_mask.any():
        root_row = feeder_rows[root_mask].iloc[0]
        root_cabinet = root_row["node2"]
        if root_cabinet not in cabinet_bus:
            # root cabinet wasn't in cable_rows (degenerate feeder with only root edge)
            root_bus_idx = pp.create_bus(net, vn_kv=_VN_KV, name=root_cabinet, type="b")
            cabinet_bus[root_cabinet] = root_bus_idx
        pp.create_line_from_parameters(
            net,
            from_bus=slack_bus,
            to_bus=cabinet_bus[root_cabinet],
            length_km=root_row["cable_length"] / 1000.0,
            r_ohm_per_km=root_row["resistance"],
            x_ohm_per_km=root_row["reactance"],
            c_nf_per_km=0.0,
            max_i_ka=root_row["cable_capacity"] / 1000.0,
            name="feeder_root_link",
        )
    else:
        # No LvFeeder root row found — fall back to connecting slack to the first node
        root_node = cable_rows.iloc[0]["node1"]
        warnings.warn(
            f"Feeder '{feeder_id}': no 'LvFeeder.*' root row found; "
            f"connecting slack to '{root_node}' via a short stub line.",
            stacklevel=2,
        )
        pp.create_line_from_parameters(
            net,
            from_bus=slack_bus,
            to_bus=cabinet_bus[root_node],
            length_km=1e-6,
            r_ohm_per_km=0.001,
            x_ohm_per_km=0.001,
            c_nf_per_km=0.0,
            max_i_ka=10.0,
            name="feeder_root_link",
        )

    # --- Loads (one per meter) ---
    feeder_meters = {
        meter_id: meta
        for meter_id, meta in mapping.items()
        if meta.get("Feeder ID") == feeder_id
    }

    meter_to_load_index: dict[str, int] = {}
    for meter_id, meta in feeder_meters.items():
        cabinet_label = f"Cabinet.{meta['Cabinet ID']}"
        if cabinet_label not in cabinet_bus:
            warnings.warn(
                f"Meter {meter_id}: cabinet '{cabinet_label}' not found in feeder "
                f"'{feeder_id}' graph — skipping",
                stacklevel=2,
            )
            continue
        load_idx = pp.create_load(
            net,
            bus=cabinet_bus[cabinet_label],
            p_mw=0.0,
            q_mvar=0.0,
            name=f"meter_{meter_id}",
        )
        meter_to_load_index[meter_id] = load_idx

    return net, meter_to_load_index
