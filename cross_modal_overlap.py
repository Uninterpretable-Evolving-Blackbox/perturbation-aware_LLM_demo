"""
Cross-modal gene overlap analysis for GSoC proposal.
Queries MaveDB and Harmonizome APIs live to compute overlaps between
CRISPR (DepMap), MAVE (MaveDB), and scPerturb-seq gene lists.

Usage: python cross_modal_overlap.py
"""

import json
import os
import re
import sys

import requests


def fetch_mavedb_genes():
    """Fetch all unique target gene names from MaveDB mapped-genes endpoint."""
    print("Fetching MaveDB target genes...")
    resp = requests.get(
        "https://api.mavedb.org/api/v1/score-sets/mapped-genes", timeout=60)
    resp.raise_for_status()
    data = resp.json()  # dict: urn -> list of gene names

    genes = set()
    for urn, info in data.items():
        if isinstance(info, list):
            for item in info:
                if isinstance(item, str):
                    genes.add(item)
        elif isinstance(info, str):
            genes.add(info)
    print(f"  MaveDB: {len(genes)} unique target genes")
    return genes


def fetch_harmonizome_genes(dataset_name, label):
    """Fetch perturbation target gene names from a Harmonizome dataset."""
    print(f"Fetching {label} from Harmonizome...")
    url = (f"https://maayanlab.cloud/Harmonizome/api/1.0/dataset/"
           f"{dataset_name}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    gene_sets = data.get("geneSets", [])

    genes = set()
    for gs in gene_sets:
        href = gs.get("href", "")
        m = re.search(r"gene_set/\d+_([A-Z][A-Z0-9]+)_", href)
        if m:
            genes.add(m.group(1))

    # Remove non-targeting controls
    genes.discard("NON")
    genes.discard("TARGETING")
    print(f"  {label}: {len(genes)} perturbation target genes")
    return genes


def main():
    # ------------------------------------------------------------------
    # 1. Fetch gene lists from live APIs
    # ------------------------------------------------------------------
    mavedb_genes = fetch_mavedb_genes()

    k562_gw = fetch_harmonizome_genes(
        "Replogle+et+al.,+Cell,+2022+K562+Genome-wide+Perturb-seq+"
        "Gene+Perturbation+Signatures",
        "Replogle K562 Genome-wide")

    k562_ess = fetch_harmonizome_genes(
        "Replogle+et+al.,+Cell,+2022+K562+Essential+Perturb-seq+"
        "Gene+Perturbation+Signatures",
        "Replogle K562 Essential")

    rpe1_ess = fetch_harmonizome_genes(
        "Replogle+et+al.,+Cell,+2022+RPE1+Essential+Perturb-seq+"
        "Gene+Perturbation+Signatures",
        "Replogle RPE1 Essential")

    # Norman 2019: 112 single-gene CRISPRa targets in K562.
    # Gene list not available from any API — the 112 hit genes are in
    # Table S1 of Norman et al. (2019) Science 365:786-793.
    # We do NOT fabricate this list. We note the count and compute
    # overlaps only with datasets we can verify.
    norman_count = 112  # from paper

    # DepMap: ~17,386 genes in Chronos_Combined (25Q3)
    # Covers essentially all protein-coding genes.
    depmap_count = 17386

    # ------------------------------------------------------------------
    # 2. Compute overlaps
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("CROSS-MODAL GENE OVERLAP ANALYSIS")
    print("=" * 65)
    print()
    print("Sources (live API queries):")
    print(f"  MaveDB target genes:           {len(mavedb_genes):>6,d}")
    print(f"  DepMap CRISPR genes:           ~{depmap_count:>5,d}  (from docs)")
    print(f"  Replogle K562 genome-wide:      {len(k562_gw):>6,d}  (Harmonizome)")
    print(f"  Replogle K562 essential:        {len(k562_ess):>6,d}  (Harmonizome)")
    print(f"  Replogle RPE1 essential:        {len(rpe1_ess):>6,d}  (Harmonizome)")
    print(f"  Norman 2019 CRISPRa:            {norman_count:>6,d}  (from paper)")

    # DEPTH: K562 genome-wide only
    depth_genes = k562_gw
    depth_mavedb = depth_genes & mavedb_genes

    # BREADTH: K562 essential + RPE1 essential (verified gene lists)
    # We exclude Norman because we can't verify the gene list
    breadth_genes = k562_ess | rpe1_ess
    breadth_mavedb = breadth_genes & mavedb_genes

    # Since DepMap covers ~87% of protein-coding genes,
    # scPerturb ∩ DepMap ≈ scPerturb total
    # Three-way ≈ scPerturb ∩ MaveDB (since DepMap is near-complete)

    print()
    print(f"{'Metric':<50s} {'Depth':>7s} {'Breadth':>7s}")
    print(f"{'':50s} {'K562GW':>7s} {'K562+R':>7s}")
    print("-" * 65)
    print(f"{'Total perturbation targets':50s} {len(depth_genes):>7,d} {len(breadth_genes):>7,d}")
    print(f"{'Cell lines':50s} {'1':>7s} {'2':>7s}")
    print()
    print(f"{'scPerturb ∩ DepMap (est. ~87% coverage)':50s} {int(len(depth_genes)*0.87):>7,d} {int(len(breadth_genes)*0.87):>7,d}")
    print(f"{'scPerturb ∩ MaveDB':50s} {len(depth_mavedb):>7,d} {len(breadth_mavedb):>7,d}")
    print(f"{'Three-way anchors (scPerturb ∩ MaveDB ∩ DepMap)':50s} {len(depth_mavedb):>7,d} {len(breadth_mavedb):>7,d}")
    print(f"  (≈ scPerturb ∩ MaveDB, since DepMap is near-complete)")

    print()
    print(f"THREE-WAY ANCHOR GENES — Depth ({len(depth_mavedb)}):")
    for g in sorted(depth_mavedb):
        print(f"  {g}")

    print()
    print(f"THREE-WAY ANCHOR GENES — Breadth ({len(breadth_mavedb)}):")
    for g in sorted(breadth_mavedb):
        print(f"  {g}")

    # Breadth-only anchors (in breadth but not depth)
    breadth_only = breadth_mavedb - depth_mavedb
    if breadth_only:
        print()
        print(f"Anchors gained by adding RPE1 ({len(breadth_only)}):")
        for g in sorted(breadth_only):
            print(f"  {g}")

    # Demo's 10 genes
    demo_genes = {"TP53", "BRCA1", "KRAS", "EGFR", "MYC", "PTEN",
                  "RB1", "BRAF", "PIK3CA", "APC"}
    demo_in_depth = demo_genes & depth_genes
    demo_in_mavedb = demo_genes & mavedb_genes
    demo_three_way = demo_genes & depth_genes & mavedb_genes

    print()
    print("OUR DEMO's 10 DepMap GENES:")
    print(f"  In K562 genome-wide: {sorted(demo_in_depth)} ({len(demo_in_depth)}/10)")
    print(f"  In MaveDB:           {sorted(demo_in_mavedb)} ({len(demo_in_mavedb)}/10)")
    print(f"  Three-way anchors:   {sorted(demo_three_way)} ({len(demo_three_way)}/10)")

    # ------------------------------------------------------------------
    # 3. Save results
    # ------------------------------------------------------------------
    results = {
        "sources": {
            "mavedb": {"count": len(mavedb_genes), "method": "live API query"},
            "depmap": {"count": depmap_count, "method": "from documentation"},
            "replogle_k562_gw": {"count": len(k562_gw), "method": "Harmonizome API"},
            "replogle_k562_ess": {"count": len(k562_ess), "method": "Harmonizome API"},
            "replogle_rpe1_ess": {"count": len(rpe1_ess), "method": "Harmonizome API"},
            "norman_2019": {"count": norman_count, "method": "from paper (not verified)"},
        },
        "depth_k562_gw": {
            "total_targets": len(depth_genes),
            "overlap_mavedb": len(depth_mavedb),
            "three_way_anchors": sorted(depth_mavedb),
        },
        "breadth_k562_rpe1": {
            "total_targets": len(breadth_genes),
            "overlap_mavedb": len(breadth_mavedb),
            "three_way_anchors": sorted(breadth_mavedb),
        },
        "demo_10_genes": {
            "in_k562_gw": sorted(demo_in_depth),
            "in_mavedb": sorted(demo_in_mavedb),
            "three_way": sorted(demo_three_way),
        },
        "gene_lists": {
            "mavedb": sorted(mavedb_genes),
            "k562_gw": sorted(k562_gw),
            "k562_ess": sorted(k562_ess),
            "rpe1_ess": sorted(rpe1_ess),
        },
    }

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(BASE_DIR, "cross_modal_overlap.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
