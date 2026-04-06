from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "samples.json"
DEFAULT_DATA_DIR = ROOT / "data" / "processed"
DEFAULT_MODEL_DIR = ROOT / "artifacts" / "models"
DEFAULT_RESULTS_DIR = ROOT / "artifacts" / "results"

BRANCH_ALIASES = {
    "el_pt": "AnalysisElectronsAuxDyn.pt",
    "el_eta": "AnalysisElectronsAuxDyn.eta",
    "el_phi": "AnalysisElectronsAuxDyn.phi",
    "mu_pt": "AnalysisMuonsAuxDyn.pt",
    "mu_eta": "AnalysisMuonsAuxDyn.eta",
    "mu_phi": "AnalysisMuonsAuxDyn.phi",
    "jet_pt": "AnalysisJetsAuxDyn.pt",
    "jet_eta": "AnalysisJetsAuxDyn.eta",
    "jet_phi": "AnalysisJetsAuxDyn.phi",
    "jet_m": "AnalysisJetsAuxDyn.m",
    "btag_prob": "BTagging_AntiKt4EMPFlowAuxDyn.DL1dv01_pb",
    "met_name": "MET_Core_AnalysisMETAuxDyn.name",
    "met_mpx": "MET_Core_AnalysisMETAuxDyn.mpx",
    "met_mpy": "MET_Core_AnalysisMETAuxDyn.mpy",
    "event_number": "EventInfoAuxDyn.eventNumber",
    "mc_event_number": "EventInfoAuxDyn.mcEventNumber",
    "mc_channel_number": "EventInfoAuxDyn.mcChannelNumber",
    "pileup_weight": "EventInfoAuxDyn.PileupWeight_NOSYS",
    "mc_event_weights": "EventInfoAuxDyn.mcEventWeights"
}

GEV = 1000.0

MODEL_FEATURES = [
    "n_jets",
    "n_bjets",
    "lep_is_muon",
    "lep_pt",
    "lep_eta_abs",
    "lead_jet_pt",
    "sublead_jet_pt",
    "third_jet_pt",
    "fourth_jet_pt",
    "ht",
    "m3",
    "m_bb",
    "deltaR_lep_nearest_b",
    "deltaR_b1_b2",
    "jet_pt_balance"
]

BOOKKEEPING_COLUMNS = [
    "sample_key",
    "sample_label",
    "sample_role",
    "process",
    "physics_short",
    "cross_section_pb",
    "source_file_index",
    "source_uid",
    "row_uid",
    "event_number",
    "mc_event_number",
    "mc_channel_number",
    "event_weight",
    "pileup_weight",
    "mc_weight"
]
