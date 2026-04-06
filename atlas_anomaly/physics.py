from __future__ import annotations

import itertools
import math
from typing import Optional

from atlas_anomaly.constants import GEV


def decode_name(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def delta_phi(phi1: float, phi2: float) -> float:
    delta = phi1 - phi2
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    return math.sqrt((eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2)


def select_leptons(pt, eta, phi, is_muon: bool) -> list[dict]:
    selected = []
    for p, e, ph in zip(pt, eta, phi):
        p_gev = float(p) / GEV
        e_val = float(e)
        ph_val = float(ph)
        if p_gev > 30.0 and abs(e_val) < 2.47:
            selected.append(
                {
                    "pt": p_gev,
                    "eta": e_val,
                    "phi": ph_val,
                    "lep_is_muon": 1 if is_muon else 0,
                }
            )
    return selected


def select_jets(pt, eta, phi, mass, btag_prob) -> list[dict]:
    jets = []
    for p, e, ph, m, btag in zip(pt, eta, phi, mass, btag_prob):
        p_gev = float(p) / GEV
        e_val = float(e)
        if p_gev <= 30.0 or abs(e_val) >= 2.47:
            continue
        jets.append(
            {
                "pt": p_gev,
                "eta": e_val,
                "phi": float(ph),
                "m": float(m) / GEV,
                "btag_prob": float(btag),
                "is_bjet": float(btag) > 0.85,
            }
        )
    return jets


def remove_jet_electron_overlap(jets: list[dict], electrons: list[dict]) -> list[dict]:
    if not electrons:
        return jets
    cleaned = []
    for jet in jets:
        overlaps = any(
            delta_r(jet["eta"], jet["phi"], electron["eta"], electron["phi"]) < 0.4
            for electron in electrons
        )
        if not overlaps:
            cleaned.append(jet)
    return cleaned


def four_vector(obj: dict) -> tuple[float, float, float, float]:
    pt = obj["pt"]
    eta = obj["eta"]
    phi = obj["phi"]
    mass = obj["m"]
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    energy = math.sqrt(px * px + py * py + pz * pz + mass * mass)
    return energy, px, py, pz


def invariant_mass(objects: list[dict]) -> float:
    energy = 0.0
    px = 0.0
    py = 0.0
    pz = 0.0
    for obj in objects:
        e_i, px_i, py_i, pz_i = four_vector(obj)
        energy += e_i
        px += px_i
        py += py_i
        pz += pz_i
    mass2 = energy * energy - px * px - py * py - pz * pz
    return math.sqrt(max(mass2, 0.0))


def top_candidate_mass(jets: list[dict]) -> float:
    best_mass = None
    best_sum_pt = -1.0
    for triplet in itertools.combinations(jets, 3):
        if not any(jet["is_bjet"] for jet in triplet):
            continue
        total_pt = sum(jet["pt"] for jet in triplet)
        if total_pt > best_sum_pt:
            best_sum_pt = total_pt
            best_mass = invariant_mass(list(triplet))
    return best_mass if best_mass is not None else float("nan")


def bb_mass(jets: list[dict]) -> float:
    bjets = sorted([jet for jet in jets if jet["is_bjet"]], key=lambda jet: jet["btag_prob"], reverse=True)
    if len(bjets) < 2:
        return float("nan")
    return invariant_mass(bjets[:2])


def nearest_bjet_delta_r(lepton: dict, jets: list[dict]) -> float:
    bjets = [jet for jet in jets if jet["is_bjet"]]
    if not bjets:
        return float("nan")
    return min(delta_r(lepton["eta"], lepton["phi"], jet["eta"], jet["phi"]) for jet in bjets)


def top_two_bjet_delta_r(jets: list[dict]) -> float:
    bjets = sorted([jet for jet in jets if jet["is_bjet"]], key=lambda jet: jet["btag_prob"], reverse=True)
    if len(bjets) < 2:
        return float("nan")
    return delta_r(bjets[0]["eta"], bjets[0]["phi"], bjets[1]["eta"], bjets[1]["phi"])


def extract_pvsoft_met_pt(names, mpx, mpy) -> float:
    for name, px, py in zip(names, mpx, mpy):
        if decode_name(name) == "PVSoftTrkCore":
            return math.sqrt(float(px) ** 2 + float(py) ** 2) / GEV
    return float("nan")


def scalar_event_weight(mc_weights, pileup_weight) -> tuple[float, float]:
    mc_weight = 1.0
    if mc_weights is not None and len(mc_weights) > 0:
        mc_weight = float(mc_weights[0])
    pileup = float(pileup_weight) if pileup_weight is not None else 1.0
    return mc_weight, mc_weight * pileup


def build_row(event_index: int, arrays: dict, sample_key: str, sample_info: dict, file_index: int) -> Optional[dict]:
    electrons = select_leptons(
        arrays["el_pt"][event_index],
        arrays["el_eta"][event_index],
        arrays["el_phi"][event_index],
        is_muon=False,
    )
    muons = select_leptons(
        arrays["mu_pt"][event_index],
        arrays["mu_eta"][event_index],
        arrays["mu_phi"][event_index],
        is_muon=True,
    )
    jets = select_jets(
        arrays["jet_pt"][event_index],
        arrays["jet_eta"][event_index],
        arrays["jet_phi"][event_index],
        arrays["jet_m"][event_index],
        arrays["btag_prob"][event_index],
    )
    jets = remove_jet_electron_overlap(jets, electrons)

    leptons = electrons + muons
    n_bjets = sum(jet["is_bjet"] for jet in jets)
    if len(leptons) != 1 or len(jets) < 4 or n_bjets < 2:
        return None

    lepton = leptons[0]
    sorted_jets = sorted(jets, key=lambda jet: jet["pt"], reverse=True)
    ht = lepton["pt"] + sum(jet["pt"] for jet in sorted_jets)
    lead_jet_pt = sorted_jets[0]["pt"]
    sublead_jet_pt = sorted_jets[1]["pt"]
    third_jet_pt = sorted_jets[2]["pt"]
    fourth_jet_pt = sorted_jets[3]["pt"]
    mc_weight, event_weight = scalar_event_weight(
        arrays["mc_event_weights"][event_index],
        arrays["pileup_weight"][event_index],
    )

    event_number = int(arrays["event_number"][event_index])
    row_uid = f"{sample_key}:{file_index}:{event_number}"

    return {
        "sample_key": sample_key,
        "sample_label": sample_info["sample_label"],
        "sample_role": sample_info["sample_role"],
        "process": sample_info["process"],
        "physics_short": sample_info["physics_short"],
        "cross_section_pb": sample_info["cross_section_pb"],
        "source_file_index": file_index,
        "source_uid": f"{sample_key}:{file_index}",
        "row_uid": row_uid,
        "event_number": event_number,
        "mc_event_number": int(arrays["mc_event_number"][event_index]),
        "mc_channel_number": int(arrays["mc_channel_number"][event_index]),
        "mc_weight": mc_weight,
        "pileup_weight": float(arrays["pileup_weight"][event_index]),
        "event_weight": event_weight,
        "n_jets": len(sorted_jets),
        "n_bjets": int(n_bjets),
        "lep_is_muon": int(lepton["lep_is_muon"]),
        "lep_pt": lepton["pt"],
        "lep_eta_abs": abs(lepton["eta"]),
        "lead_jet_pt": lead_jet_pt,
        "sublead_jet_pt": sublead_jet_pt,
        "third_jet_pt": third_jet_pt,
        "fourth_jet_pt": fourth_jet_pt,
        "ht": ht,
        "m3": top_candidate_mass(sorted_jets),
        "m_bb": bb_mass(sorted_jets),
        "deltaR_lep_nearest_b": nearest_bjet_delta_r(lepton, sorted_jets),
        "deltaR_b1_b2": top_two_bjet_delta_r(sorted_jets),
        "jet_pt_balance": lead_jet_pt / ht if ht > 0 else float("nan"),
        "pvsofttrkcore_met_pt": extract_pvsoft_met_pt(
            arrays["met_name"][event_index],
            arrays["met_mpx"][event_index],
            arrays["met_mpy"][event_index],
        ),
    }
