#!/usr/bin/env python3
"""AlphaFold3 ZIP → Standalone HTML Report Generator.

Takes an AF3 output ZIP file and produces a single beautiful standalone HTML
report with embedded PAE heatmaps, confidence scores, and interface analysis.

Usage:
    python af3_report.py fold_tigd4_dimer.zip
    python af3_report.py fold_tigd4_dimer.zip -o report.html
    python af3_report.py fold_tigd4_dimer.zip --contact-dist 6.0

Output:
    A single standalone HTML file with all images embedded as base64 data URIs.
"""

import argparse
import base64
import io
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Constants ──────────────────────────────────────────────────────────────────

CONTACT_DISTANCE = 8.0  # Angstroms
HIGH_CONFIDENCE_PLDDT = 70.0
HIGH_PAE = 10.0  # Angstroms, above = low confidence

PAE_CMAP = LinearSegmentedColormap.from_list(
    "pae",
    ["#0a2463", "#3e92cc", "#88d498", "#f6f740", "#f77f00", "#d62828"],
    N=256,
)

# Three-letter to one-letter amino acid mapping
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

DNA_BASES = {"DA", "DT", "DG", "DC", "A", "T", "G", "C", "U"}
RNA_BASES = {"A", "U", "G", "C"}


# ── CIF Parser ─────────────────────────────────────────────────────────────────

def parse_cif_atoms(cif_path):
    """Parse mmCIF ATOM records into structured arrays.

    Returns dict with keys: chain_id, res_name, res_id, atom_name,
    coords (Nx3), plddt, element
    """
    chains = []
    res_names = []
    res_ids = []
    atom_names = []
    coords = []
    plddts = []
    elements = []

    with open(cif_path) as f:
        in_atom_block = False
        for line in f:
            if line.startswith("_atom_site."):
                in_atom_block = True
                continue
            if in_atom_block and not line.startswith(("ATOM", "HETATM")):
                if line.strip() == "#" or line.startswith("_") or line.startswith("loop_"):
                    in_atom_block = False
                continue
            if not in_atom_block:
                continue

            parts = line.split()
            if len(parts) < 18:
                continue

            element = parts[2]
            if element == "H":
                continue

            chains.append(parts[6])
            res_names.append(parts[5])
            res_ids.append(int(parts[8]))
            atom_names.append(parts[3])
            coords.append([float(parts[10]), float(parts[11]), float(parts[12])])
            plddts.append(float(parts[14]))
            elements.append(element)

    return {
        "chain_id": np.array(chains),
        "res_name": np.array(res_names),
        "res_id": np.array(res_ids),
        "atom_name": np.array(atom_names),
        "coords": np.array(coords) if coords else np.empty((0, 3)),
        "plddt": np.array(plddts),
        "element": np.array(elements),
    }


# ── AF3 Data Loaders ──────────────────────────────────────────────────────────

def load_full_data(json_path):
    """Load AF3 full_data JSON (PAE matrix, contact_probs, pLDDT)."""
    with open(json_path) as f:
        data = json.load(f)
    return {
        "pae": np.array(data["pae"]),
        "contact_probs": np.array(data.get("contact_probs", [])),
        "token_chain_ids": data["token_chain_ids"],
        "token_res_ids": data["token_res_ids"],
        "atom_plddts": data.get("atom_plddts", []),
        "atom_chain_ids": data.get("atom_chain_ids", []),
    }


def load_summary_confidences(json_path):
    """Load AF3 summary confidence scores."""
    with open(json_path) as f:
        return json.load(f)


def load_job_request(json_path):
    """Load AF3 job request for metadata.

    AF3 job_request.json can be either a list (with one element) or a dict.
    """
    with open(json_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data[0] if data else {}
    return data


# ── Chain Info Extraction ──────────────────────────────────────────────────────

def classify_chain(res_names):
    """Classify a chain as protein, DNA, RNA, or ligand based on residue names."""
    unique = set(res_names)
    if unique & {"DA", "DT", "DG", "DC"}:
        return "DNA"
    if unique & {"A", "U", "G", "C"} and not (unique & set(AA3_TO_1.keys())):
        return "RNA"
    if unique & set(AA3_TO_1.keys()):
        return "protein"
    return "ligand"


def get_chain_info(atoms):
    """Extract chain-level summary: type, length, sequence."""
    unique_chains = sorted(set(atoms["chain_id"]))
    info = {}
    for chain in unique_chains:
        mask = atoms["chain_id"] == chain
        res_names = atoms["res_name"][mask]
        res_ids = atoms["res_id"][mask]

        unique_residues = []
        seen = set()
        for rn, ri in zip(res_names, res_ids):
            key = (rn, ri)
            if key not in seen:
                seen.add(key)
                unique_residues.append(rn)

        chain_type = classify_chain(unique_residues)
        plddts = atoms["plddt"][mask]

        info[chain] = {
            "type": chain_type,
            "length": len(unique_residues),
            "mean_plddt": round(float(np.mean(plddts)), 1),
            "residue_names": unique_residues,
        }
    return info


# ── Interface Detection ────────────────────────────────────────────────────────

def find_interfaces(atoms, contact_dist=CONTACT_DISTANCE):
    """Find inter-chain contacts using KDTree."""
    unique_chains = sorted(set(atoms["chain_id"]))
    if len(unique_chains) < 2:
        return []

    tree = cKDTree(atoms["coords"])
    pairs = tree.query_pairs(r=contact_dist)

    interfaces = {}
    for i, j in pairs:
        ci, cj = atoms["chain_id"][i], atoms["chain_id"][j]
        if ci == cj:
            continue
        key = tuple(sorted([ci, cj]))
        if key not in interfaces:
            interfaces[key] = []
        interfaces[key].append((i, j))

    results = []
    for (chain_a, chain_b), contact_pairs in interfaces.items():
        residues_a = set()
        residues_b = set()
        contacts_detail = []

        for i, j in contact_pairs:
            ci, cj = atoms["chain_id"][i], atoms["chain_id"][j]
            if ci == chain_b:
                i, j = j, i

            ri = int(atoms["res_id"][i])
            rj = int(atoms["res_id"][j])
            dist = np.linalg.norm(atoms["coords"][i] - atoms["coords"][j])

            residues_a.add(ri)
            residues_b.add(rj)
            contacts_detail.append({
                "res_a": ri,
                "atom_a": str(atoms["atom_name"][i]),
                "resname_a": str(atoms["res_name"][i]),
                "plddt_a": float(atoms["plddt"][i]),
                "res_b": rj,
                "atom_b": str(atoms["atom_name"][j]),
                "resname_b": str(atoms["res_name"][j]),
                "plddt_b": float(atoms["plddt"][j]),
                "distance": round(dist, 2),
            })

        avg_plddt_a = np.mean([c["plddt_a"] for c in contacts_detail])
        avg_plddt_b = np.mean([c["plddt_b"] for c in contacts_detail])
        high_conf = sum(
            1 for c in contacts_detail
            if c["plddt_a"] >= HIGH_CONFIDENCE_PLDDT and c["plddt_b"] >= HIGH_CONFIDENCE_PLDDT
        )

        results.append({
            "chain_a": chain_a,
            "chain_b": chain_b,
            "residues_a": sorted(residues_a),
            "residues_b": sorted(residues_b),
            "n_residues_a": len(residues_a),
            "n_residues_b": len(residues_b),
            "n_atom_contacts": len(contacts_detail),
            "avg_plddt_a": round(float(avg_plddt_a), 1),
            "avg_plddt_b": round(float(avg_plddt_b), 1),
            "high_conf_pct": round(high_conf / len(contacts_detail) * 100, 1) if contacts_detail else 0,
            "contacts": contacts_detail,
        })

    return results


def add_pae_to_interfaces(interfaces, full_data):
    """Cross-reference interface residues with PAE matrix."""
    token_chains = full_data["token_chain_ids"]
    token_res = full_data["token_res_ids"]
    pae = full_data["pae"]

    token_idx = {}
    for idx, (c, r) in enumerate(zip(token_chains, token_res)):
        token_idx[(c, r)] = idx

    for iface in interfaces:
        ca, cb = iface["chain_a"], iface["chain_b"]
        pae_values = []
        for ra in iface["residues_a"]:
            for rb in iface["residues_b"]:
                ia = token_idx.get((ca, ra))
                ib = token_idx.get((cb, rb))
                if ia is not None and ib is not None:
                    pae_values.append(float(pae[ia][ib]))
                    pae_values.append(float(pae[ib][ia]))

        if pae_values:
            iface["avg_pae"] = round(np.mean(pae_values), 2)
            iface["min_pae"] = round(np.min(pae_values), 2)
            iface["max_pae"] = round(np.max(pae_values), 2)
            iface["pae_below_10"] = round(
                sum(1 for v in pae_values if v < HIGH_PAE) / len(pae_values) * 100, 1
            )
        else:
            iface["avg_pae"] = None
            iface["min_pae"] = None
            iface["max_pae"] = None
            iface["pae_below_10"] = None

    return interfaces


# ── PAE Heatmap Generation (in-memory → base64) ───────────────────────────────

def find_chain_boundaries(token_chain_ids):
    """Find start/end indices for each chain in token list."""
    boundaries = {}
    current_chain = token_chain_ids[0]
    start = 0
    for i, chain in enumerate(token_chain_ids):
        if chain != current_chain:
            boundaries[current_chain] = (start, i)
            current_chain = chain
            start = i
    boundaries[current_chain] = (start, len(token_chain_ids))
    return boundaries


def pae_heatmap_to_base64(pae, token_chain_ids, title):
    """Generate a full PAE heatmap and return as base64 PNG string."""
    boundaries = find_chain_boundaries(token_chain_ids)
    n = len(pae)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(pae, cmap=PAE_CMAP, vmin=0, vmax=30, aspect="equal",
                   interpolation="nearest")

    for chain, (start, end) in boundaries.items():
        if start > 0:
            ax.axhline(y=start - 0.5, color="white", linewidth=1, alpha=0.8)
            ax.axvline(x=start - 0.5, color="white", linewidth=1, alpha=0.8)
        mid = (start + end) / 2
        ax.text(mid, -n * 0.03, f"Chain {chain}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="white")
        ax.text(-n * 0.03, mid, f"Chain {chain}", ha="right", va="center",
                fontsize=10, fontweight="bold", color="white", rotation=90)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        label="Predicted Aligned Error (Å)")
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Scored residue", fontsize=10)
    ax.set_ylabel("Aligned residue", fontsize=10)
    ax.tick_params(labelsize=8)

    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(colors="white")
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def interface_pae_to_base64(pae, token_chain_ids, token_res_ids, chain_a, chain_b, title):
    """Generate interface PAE sub-matrix heatmap and return as base64 PNG."""
    boundaries = find_chain_boundaries(token_chain_ids)

    if chain_a not in boundaries or chain_b not in boundaries:
        return None

    sa, ea = boundaries[chain_a]
    sb, eb = boundaries[chain_b]
    sub_pae = pae[sa:ea, sb:eb]
    res_a = token_res_ids[sa:ea]
    res_b = token_res_ids[sb:eb]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(sub_pae, cmap=PAE_CMAP, vmin=0, vmax=30, aspect="auto",
                   interpolation="nearest")

    xticks = list(range(0, len(res_b), max(1, len(res_b) // 10)))
    yticks = list(range(0, len(res_a), max(1, len(res_a) // 10)))
    ax.set_xticks(xticks)
    ax.set_xticklabels([res_b[i] for i in xticks], fontsize=8)
    ax.set_yticks(yticks)
    ax.set_yticklabels([res_a[i] for i in yticks], fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="PAE (Å)")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(f"Chain {chain_b} residue", fontsize=10)
    ax.set_ylabel(f"Chain {chain_a} residue", fontsize=10)

    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(colors="white")
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")

    # Stats annotation
    mean_pae = np.mean(sub_pae)
    pct_below10 = np.mean(sub_pae < 10) * 100
    ax.text(0.02, 0.98, f"mean PAE={mean_pae:.1f}Å\n<10Å: {pct_below10:.0f}%",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", alpha=0.9))

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ── ZIP Extraction & Discovery ─────────────────────────────────────────────────

def extract_zip(zip_path, tmp_dir):
    """Extract AF3 ZIP to temp directory, return model directory path."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    # AF3 ZIPs often have a single top-level directory
    entries = list(Path(tmp_dir).iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return Path(tmp_dir)


def discover_models(model_dir):
    """Auto-discover model files in the extracted directory.

    Returns list of model indices found.
    """
    indices = []
    for f in sorted(model_dir.iterdir()):
        if f.name.endswith(".cif") and "_model_" in f.name:
            # Extract index from e.g. fold_tigd4_dimer_model_3.cif
            idx_str = f.stem.split("_model_")[-1]
            try:
                indices.append(int(idx_str))
            except ValueError:
                pass
    return sorted(indices)


# ── Main Processing Pipeline ───────────────────────────────────────────────────

def process_all_models(model_dir, contact_dist=CONTACT_DISTANCE):
    """Process all models in a directory, return structured data for HTML."""
    name = model_dir.name
    model_indices = discover_models(model_dir)

    if not model_indices:
        print(f"ERROR: No model CIF files found in {model_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(model_indices)} models: {model_indices}")

    # Load job request
    job_request = None
    job_req_path = model_dir / f"{name}_job_request.json"
    if job_req_path.exists():
        job_request = load_job_request(str(job_req_path))

    models = []
    for idx in model_indices:
        cif_path = model_dir / f"{name}_model_{idx}.cif"
        full_data_path = model_dir / f"{name}_full_data_{idx}.json"
        conf_path = model_dir / f"{name}_summary_confidences_{idx}.json"

        if not cif_path.exists():
            print(f"  Skipping model_{idx}: CIF not found")
            continue

        print(f"  Processing model_{idx}...")

        atoms = parse_cif_atoms(str(cif_path))
        chain_info = get_chain_info(atoms)
        summary_conf = load_summary_confidences(str(conf_path)) if conf_path.exists() else {}

        interfaces = find_interfaces(atoms, contact_dist)

        full_data = None
        pae_full_b64 = None
        pae_interface_b64s = []

        if full_data_path.exists():
            full_data = load_full_data(str(full_data_path))
            interfaces = add_pae_to_interfaces(interfaces, full_data)

            # Generate full PAE heatmap
            pae_full_b64 = pae_heatmap_to_base64(
                full_data["pae"],
                full_data["token_chain_ids"],
                f"Model {idx} — Full PAE",
            )

            # Generate per-interface PAE heatmaps
            chains = sorted(chain_info.keys())
            for i, ca in enumerate(chains):
                for cb in chains[i + 1:]:
                    b64 = interface_pae_to_base64(
                        full_data["pae"],
                        full_data["token_chain_ids"],
                        full_data["token_res_ids"],
                        ca, cb,
                        f"Model {idx} — Interface {ca}–{cb}",
                    )
                    if b64:
                        pae_interface_b64s.append({
                            "chain_a": ca,
                            "chain_b": cb,
                            "b64": b64,
                        })

        # Strip heavy contacts detail for summary
        interfaces_light = []
        for iface in interfaces:
            light = {k: v for k, v in iface.items() if k != "contacts"}
            interfaces_light.append(light)

        model_data = {
            "idx": idx,
            "confidence": summary_conf,
            "chain_info": chain_info,
            "interfaces": interfaces_light,
            "pae_full_b64": pae_full_b64,
            "pae_interface_b64s": pae_interface_b64s,
        }
        models.append(model_data)

    return {
        "name": name,
        "job_request": job_request,
        "models": models,
    }


# ── HTML Report Generation ─────────────────────────────────────────────────────

def generate_html(data):
    """Generate standalone HTML report from processed data."""
    name = data["name"]
    job_request = data["job_request"]
    models = data["models"]

    # Sort by ranking_score descending
    models_sorted = sorted(
        models,
        key=lambda m: m["confidence"].get("ranking_score", 0),
        reverse=True,
    )

    best_model = models_sorted[0] if models_sorted else None
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Extract sequences from job_request
    sequences_html = ""
    if job_request:
        seqs = job_request.get("sequences", [])
        if seqs:
            rows = []
            for i, seq in enumerate(seqs):
                if "proteinChain" in seq:
                    pc = seq["proteinChain"]
                    seq_str = pc.get("sequence", "")
                    chain_type = "Protein"
                    length = len(seq_str)
                    desc = seq_str[:60] + "..." if len(seq_str) > 60 else seq_str
                elif "dnaSequence" in seq:
                    ds = seq["dnaSequence"]
                    seq_str = ds.get("sequence", "")
                    chain_type = "DNA"
                    length = len(seq_str)
                    desc = seq_str[:60] + "..." if len(seq_str) > 60 else seq_str
                elif "rnaSequence" in seq:
                    rs = seq["rnaSequence"]
                    seq_str = rs.get("sequence", "")
                    chain_type = "RNA"
                    length = len(seq_str)
                    desc = seq_str[:60] + "..." if len(seq_str) > 60 else seq_str
                elif "ligand" in seq:
                    lig = seq["ligand"]
                    chain_type = "Ligand"
                    length = "-"
                    desc = lig.get("ccdCodes", ["?"])[0] if "ccdCodes" in lig else "custom"
                else:
                    chain_type = "Unknown"
                    length = "-"
                    desc = str(seq)[:60]

                rows.append(f"""<tr>
                    <td>{i + 1}</td>
                    <td><span class="badge badge-{chain_type.lower()}">{chain_type}</span></td>
                    <td>{length}</td>
                    <td class="seq-preview">{escape(str(desc))}</td>
                </tr>""")
            sequences_html = "\n".join(rows)

    # Model name from job_request
    model_name = name
    if job_request:
        model_name = job_request.get("name", name)

    # Confidence table
    confidence_rows = []
    for m in models_sorted:
        conf = m["confidence"]
        is_best = m is best_model
        row_class = "best-row" if is_best else ""
        badge = ' <span class="badge badge-best">BEST</span>' if is_best else ""
        clash_icon = "&#x26A0;" if conf.get("has_clash") else "&#x2713;"
        clash_class = "clash-yes" if conf.get("has_clash") else "clash-no"

        confidence_rows.append(f"""<tr class="{row_class}">
            <td>Model {m['idx']}{badge}</td>
            <td class="mono">{conf.get('ranking_score', '-')}</td>
            <td class="mono">{conf.get('iptm', '-')}</td>
            <td class="mono">{conf.get('ptm', '-')}</td>
            <td class="mono">{conf.get('fraction_disordered', '-')}</td>
            <td class="{clash_class}">{clash_icon}</td>
        </tr>""")

    # Interface summary (from best model)
    interface_summary_rows = ""
    if best_model and best_model["interfaces"]:
        rows = []
        for iface in best_model["interfaces"]:
            pae_cell = f"{iface['avg_pae']:.1f}" if iface.get("avg_pae") is not None else "-"
            pae_class = ""
            if iface.get("avg_pae") is not None:
                if iface["avg_pae"] < 5:
                    pae_class = "pae-good"
                elif iface["avg_pae"] < 10:
                    pae_class = "pae-ok"
                else:
                    pae_class = "pae-bad"

            pae_pct = f"{iface['pae_below_10']:.0f}%" if iface.get("pae_below_10") is not None else "-"

            rows.append(f"""<tr>
                <td>{iface['chain_a']}–{iface['chain_b']}</td>
                <td class="mono">{iface['n_residues_a']}</td>
                <td class="mono">{iface['n_residues_b']}</td>
                <td class="mono">{iface['n_atom_contacts']}</td>
                <td class="mono">{iface['avg_plddt_a']:.1f}</td>
                <td class="mono">{iface['avg_plddt_b']:.1f}</td>
                <td class="mono {pae_class}">{pae_cell}</td>
                <td class="mono">{pae_pct}</td>
                <td class="mono">{iface['high_conf_pct']:.0f}%</td>
            </tr>""")
        interface_summary_rows = "\n".join(rows)

    # PAE heatmaps section
    pae_sections = []
    for i, m in enumerate(models_sorted):
        if not m.get("pae_full_b64"):
            continue
        collapsed = "collapsed" if i > 0 else ""
        hidden = "hidden" if i > 0 else ""
        pae_sections.append(f"""
        <div class="pae-model">
            <h3 class="collapsible {collapsed}" onclick="toggleCollapse(this)">
                <span class="arrow">{'▶' if i > 0 else '▼'}</span>
                Model {m['idx']}
                <span class="confidence-mini">
                    ranking={m['confidence'].get('ranking_score', '-')}
                    ipTM={m['confidence'].get('iptm', '-')}
                </span>
            </h3>
            <div class="collapsible-content {hidden}">
                <img src="data:image/png;base64,{m['pae_full_b64']}"
                     alt="PAE heatmap model {m['idx']}" class="pae-img">
                {''.join(f'<img src="data:image/png;base64,{ip["b64"]}" alt="Interface {ip["chain_a"]}-{ip["chain_b"]}" class="pae-img pae-interface-img">' for ip in m.get("pae_interface_b64s", []))}
            </div>
        </div>""")

    # Per-model details
    detail_sections = []
    for i, m in enumerate(models_sorted):
        if not m["interfaces"]:
            continue
        collapsed = "collapsed" if i > 0 else ""
        hidden = "hidden" if i > 0 else ""

        iface_rows = []
        for iface in m["interfaces"]:
            pae_cell = f"{iface['avg_pae']:.1f}" if iface.get("avg_pae") is not None else "-"
            pae_class = ""
            if iface.get("avg_pae") is not None:
                pae_class = "pae-good" if iface["avg_pae"] < 5 else ("pae-ok" if iface["avg_pae"] < 10 else "pae-bad")

            # Residue ranges
            def fmt_ranges(residues):
                if not residues:
                    return "-"
                ranges = []
                start = residues[0]
                end = residues[0]
                for r in residues[1:]:
                    if r == end + 1:
                        end = r
                    else:
                        ranges.append(f"{start}-{end}" if start != end else str(start))
                        start = end = r
                ranges.append(f"{start}-{end}" if start != end else str(start))
                return ", ".join(ranges)

            res_a_str = fmt_ranges(iface["residues_a"])
            res_b_str = fmt_ranges(iface["residues_b"])

            iface_rows.append(f"""<tr>
                <td>{iface['chain_a']}–{iface['chain_b']}</td>
                <td class="mono">{iface['n_atom_contacts']}</td>
                <td class="mono {pae_class}">{pae_cell}</td>
                <td class="mono">{iface['avg_plddt_a']:.1f} / {iface['avg_plddt_b']:.1f}</td>
                <td class="mono">{iface['high_conf_pct']:.0f}%</td>
                <td class="residue-range">{res_a_str}</td>
                <td class="residue-range">{res_b_str}</td>
            </tr>""")

        # Chain info table
        chain_rows = []
        for chain_id, ci in m["chain_info"].items():
            chain_rows.append(f"""<tr>
                <td>{chain_id}</td>
                <td><span class="badge badge-{ci['type']}">{ci['type']}</span></td>
                <td class="mono">{ci['length']}</td>
                <td class="mono">{ci['mean_plddt']}</td>
            </tr>""")

        detail_sections.append(f"""
        <div class="model-detail">
            <h3 class="collapsible {collapsed}" onclick="toggleCollapse(this)">
                <span class="arrow">{'▶' if i > 0 else '▼'}</span>
                Model {m['idx']} Details
                <span class="confidence-mini">
                    ranking={m['confidence'].get('ranking_score', '-')}
                </span>
            </h3>
            <div class="collapsible-content {hidden}">
                <h4>Chains</h4>
                <table class="data-table">
                    <thead><tr>
                        <th>Chain</th><th>Type</th><th>Residues</th><th>Mean pLDDT</th>
                    </tr></thead>
                    <tbody>{''.join(chain_rows)}</tbody>
                </table>

                <h4>Interfaces</h4>
                <table class="data-table">
                    <thead><tr>
                        <th>Interface</th><th>Contacts</th><th>Avg PAE</th>
                        <th>pLDDT (A/B)</th><th>High-conf %</th>
                        <th>Residues A</th><th>Residues B</th>
                    </tr></thead>
                    <tbody>{''.join(iface_rows)}</tbody>
                </table>
            </div>
        </div>""")

    # Check if this is single-chain (no interfaces)
    is_single_chain = not best_model or not best_model["interfaces"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AF3 Report: {escape(model_name)}</title>
<style>
:root {{
    --bg: #0f172a;
    --card: #1e293b;
    --card-hover: #273548;
    --border: #334155;
    --text: #e2e8f0;
    --text-dim: #94a3b8;
    --accent: #0891b2;
    --accent-light: #22d3ee;
    --green: #10b981;
    --yellow: #f59e0b;
    --red: #ef4444;
    --orange: #f97316;
    --purple: #a78bfa;
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
    max-width: 1200px;
    margin: 0 auto;
}}

h1 {{
    font-size: 1.8rem;
    color: var(--accent-light);
    margin-bottom: 0.25rem;
}}
h2 {{
    font-size: 1.3rem;
    color: var(--accent);
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}}
h3 {{
    font-size: 1.1rem;
    color: var(--text);
    margin: 1rem 0 0.5rem;
}}
h4 {{
    font-size: 0.95rem;
    color: var(--text-dim);
    margin: 1rem 0 0.5rem;
}}

.header {{
    margin-bottom: 2rem;
}}
.header .subtitle {{
    color: var(--text-dim);
    font-size: 0.9rem;
}}

.card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}}

.data-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 0.5rem 0;
    font-size: 0.88rem;
}}
.data-table th {{
    background: rgba(8, 145, 178, 0.15);
    color: var(--accent-light);
    padding: 0.6rem 0.8rem;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid var(--border);
    white-space: nowrap;
}}
.data-table td {{
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
}}
.data-table tr:hover {{
    background: var(--card-hover);
}}
.data-table .best-row {{
    background: rgba(16, 185, 129, 0.08);
    border-left: 3px solid var(--green);
}}
.data-table .best-row:hover {{
    background: rgba(16, 185, 129, 0.14);
}}

.mono {{
    font-family: "SF Mono", "Fira Code", "Cascadia Code", monospace;
    font-size: 0.85rem;
}}

.seq-preview {{
    font-family: "SF Mono", "Fira Code", monospace;
    font-size: 0.75rem;
    color: var(--text-dim);
    word-break: break-all;
    max-width: 400px;
}}

.residue-range {{
    font-family: "SF Mono", monospace;
    font-size: 0.78rem;
    color: var(--text-dim);
    max-width: 200px;
    word-break: break-all;
}}

.badge {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
}}
.badge-protein {{ background: rgba(167, 139, 250, 0.2); color: var(--purple); }}
.badge-dna {{ background: rgba(8, 145, 178, 0.2); color: var(--accent-light); }}
.badge-rna {{ background: rgba(249, 115, 22, 0.2); color: var(--orange); }}
.badge-ligand {{ background: rgba(245, 158, 11, 0.2); color: var(--yellow); }}
.badge-unknown {{ background: rgba(148, 163, 184, 0.2); color: var(--text-dim); }}
.badge-best {{ background: rgba(16, 185, 129, 0.3); color: var(--green); }}

.pae-good {{ color: var(--green); font-weight: 600; }}
.pae-ok {{ color: var(--yellow); }}
.pae-bad {{ color: var(--red); }}

.clash-yes {{ color: var(--red); }}
.clash-no {{ color: var(--green); }}

.pae-img {{
    max-width: 100%;
    border-radius: 6px;
    border: 1px solid var(--border);
    margin: 0.5rem 0;
}}
.pae-interface-img {{
    max-width: 48%;
    display: inline-block;
    margin: 0.5rem 0.5%;
}}

.collapsible {{
    cursor: pointer;
    user-select: none;
    padding: 0.5rem 0;
    transition: color 0.2s;
}}
.collapsible:hover {{
    color: var(--accent-light);
}}
.collapsible .arrow {{
    display: inline-block;
    width: 1.2rem;
    font-size: 0.8rem;
    transition: transform 0.2s;
}}
.collapsible .confidence-mini {{
    float: right;
    font-size: 0.8rem;
    color: var(--text-dim);
    font-weight: normal;
    font-family: monospace;
}}

.hidden {{
    display: none;
}}

.footer {{
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    color: var(--text-dim);
    font-size: 0.8rem;
    text-align: center;
}}

@media (max-width: 768px) {{
    body {{ padding: 1rem; }}
    .pae-interface-img {{ max-width: 100%; display: block; }}
    .data-table {{ font-size: 0.8rem; }}
    .data-table th, .data-table td {{ padding: 0.4rem; }}
}}
</style>
</head>
<body>

<div class="header">
    <h1>{escape(model_name)}</h1>
    <div class="subtitle">
        AlphaFold3 Structure Report &middot; Generated {now}
        {f' &middot; Seeds: {", ".join(str(s) for s in job_request.get("modelSeeds", []))}' if job_request and job_request.get("modelSeeds") else ''}
    </div>
</div>

{'<h2>Input Sequences</h2>' + '<div class="card"><table class="data-table"><thead><tr><th>#</th><th>Type</th><th>Length</th><th>Sequence</th></tr></thead><tbody>' + sequences_html + '</tbody></table></div>' if sequences_html else ''}

<h2>Confidence Overview</h2>
<div class="card">
    <table class="data-table">
        <thead><tr>
            <th>Model</th>
            <th>Ranking Score</th>
            <th>ipTM</th>
            <th>pTM</th>
            <th>Frac. Disordered</th>
            <th>Clash</th>
        </tr></thead>
        <tbody>
            {''.join(confidence_rows)}
        </tbody>
    </table>
</div>

{'<h2>Interface Analysis (Best Model)</h2>' + '<div class="card"><table class="data-table"><thead><tr><th>Interface</th><th>Res. A</th><th>Res. B</th><th>Atom Contacts</th><th>pLDDT A</th><th>pLDDT B</th><th>Avg PAE</th><th>PAE &lt;10Å</th><th>High-conf</th></tr></thead><tbody>' + interface_summary_rows + '</tbody></table></div>' if not is_single_chain else '<div class="card" style="color: var(--text-dim);">Single chain model — no interface analysis.</div>'}

<h2>PAE Heatmaps</h2>
{''.join(pae_sections) if pae_sections else '<div class="card" style="color: var(--text-dim);">No PAE data available.</div>'}

{'<h2>Per-Model Details</h2>' + ''.join(detail_sections) if detail_sections else ''}

<div class="footer">
    Generated by <strong>af3_report.py</strong> (alpha_fold_viewer)
    &middot; {escape(name)} &middot; {len(models)} model(s)
</div>

<script>
function toggleCollapse(el) {{
    const content = el.nextElementSibling;
    const arrow = el.querySelector('.arrow');
    if (content.classList.contains('hidden')) {{
        content.classList.remove('hidden');
        arrow.textContent = '\\u25BC';
        el.classList.remove('collapsed');
    }} else {{
        content.classList.add('hidden');
        arrow.textContent = '\\u25B6';
        el.classList.add('collapsed');
    }}
}}
</script>

</body>
</html>"""

    return html


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AlphaFold3 ZIP → Standalone HTML Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python af3_report.py fold_tigd4_dimer.zip
    python af3_report.py fold_tigd4_dimer.zip -o my_report.html
    python af3_report.py fold_tigd4_dimer.zip --contact-dist 6.0
        """,
    )
    parser.add_argument("input_zip", help="Path to AlphaFold3 output ZIP file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output HTML file (default: {zip_name}_report.html)")
    parser.add_argument("--contact-dist", type=float, default=CONTACT_DISTANCE,
                        help=f"Inter-atomic contact threshold in Angstroms (default: {CONTACT_DISTANCE})")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Keep extracted temporary files")

    args = parser.parse_args()

    zip_path = Path(args.input_zip)
    if not zip_path.exists():
        print(f"ERROR: File not found: {zip_path}", file=sys.stderr)
        sys.exit(1)
    if not zipfile.is_zipfile(str(zip_path)):
        print(f"ERROR: Not a valid ZIP file: {zip_path}", file=sys.stderr)
        sys.exit(1)

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = zip_path.with_name(zip_path.stem + "_report.html")

    print(f"Input:  {zip_path}")
    print(f"Output: {output_path}")
    print(f"Contact distance: {args.contact_dist} Å")
    print()

    # Extract ZIP
    tmp_dir = tempfile.mkdtemp(prefix="af3_report_")
    try:
        print("Extracting ZIP...")
        model_dir = extract_zip(str(zip_path), tmp_dir)
        print(f"Model directory: {model_dir.name}")
        print()

        # Process
        data = process_all_models(model_dir, args.contact_dist)

        # Generate HTML
        print("\nGenerating HTML report...")
        html = generate_html(data)

        with open(output_path, "w") as f:
            f.write(html)

        file_size = output_path.stat().st_size
        print(f"\nDone! Report saved to: {output_path}")
        print(f"File size: {file_size / 1024:.0f} KB")

    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            print(f"Temp files kept at: {tmp_dir}")


if __name__ == "__main__":
    main()
