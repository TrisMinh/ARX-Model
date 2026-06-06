from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = OUT_DIR / "results"
REDO_SRC = ROOT / "ARX_Redo_70_75" / "src"

if str(REDO_SRC) not in sys.path:
    sys.path.insert(0, str(REDO_SRC))

import mpc_model_audit as audit  # noqa: E402
import narx_clean_experiments as narx_clean  # noqa: E402
import narx_end_to_end_75 as narx_e2e  # noqa: E402


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def metric_from_arx(result: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    test = result["selected_metrics"]["test"]
    selected = result["selected_by_validation"]
    return {
        "case": "ARX robust search",
        "dataset": dataset_name,
        "family": "Linear ARX",
        "selected_model": selected["model"],
        "selection_policy": "validation robust score",
        "test_FIT_1step": test["metrics_1step"]["FIT"],
        "test_FIT_12": test["metrics_12"]["FIT"],
        "test_FIT_sim": test["metrics_sim"]["FIT"],
        "test_RMSE_sim": test["metrics_sim"]["RMSE"],
        "mpc_note": "MPC-friendly linear plant model",
    }


def metric_from_clean_narx(payload: dict[str, Any]) -> dict[str, Any]:
    selected = payload["selected_by_validation"]
    return {
        "case": "Clean NARX",
        "dataset": "current feedback data",
        "family": "Nonlinear NNARX",
        "selected_model": selected["model"],
        "selection_policy": "validation FIT_sim",
        "test_FIT_1step": selected["test_FIT_1step"],
        "test_FIT_12": selected["test_FIT_12"],
        "test_FIT_sim": selected["test_FIT_sim"],
        "test_RMSE_sim": selected["test_RMSE_sim"],
        "mpc_note": "Not plug-compatible with current linear/RLS MPC",
    }


def metric_from_e2e_narx(payload: dict[str, Any]) -> dict[str, Any]:
    selected = payload["selected_by_validation"]
    test = payload["selected_metrics"]["test"]
    return {
        "case": "Delta-NNARX identifiable protocol",
        "dataset": "identifiable protocol data",
        "family": "Nonlinear NNARX",
        "selected_model": selected["model"],
        "selection_policy": "validation robust score",
        "test_FIT_1step": test["metrics_1step"]["FIT"],
        "test_FIT_12": test["metrics_12"]["FIT"],
        "test_FIT_sim": test["metrics_sim"]["FIT"],
        "test_RMSE_sim": test["metrics_sim"]["RMSE"],
        "mpc_note": "Needs NMPC/local linearization for direct control use",
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_f):
        return "NA"
    return f"{value_f:.{digits}f}"


def write_final_summary(payload: dict[str, Any]) -> None:
    rows = payload["comparison_rows"]
    data_current = payload["arx_audit"]["data_audit"]["old_feedback_data"]
    data_new = payload["arx_audit"]["data_audit"]["new_identifiable_data"]
    old_diag = data_current["actuator_diagnostics"]
    new_diag = data_new["actuator_diagnostics"]

    lines = [
        "# Final Practical ARX/NARX Summary",
        "",
        "## Ket luan chinh",
        "",
        "- Current data khong bi missing/duplicate va sampling dung 300s, nhung la closed-loop feedback data.",
        "- Tren current data, ARX van la lua chon dung logic hon cho MPC: free-run on dinh hon NARX va plug-compatible voi MPC tuyen tinh/RLS.",
        "- NARX khong tao ra loi the ro tren current data: 1-step cao hon, nhung free-run thap hon ARX.",
        "- Tren identifiable protocol data, ARX va NARX deu dat moc >75 FIT_sim; chenh lech free-run giua NARX va ARX rat nho.",
        "- Huong cai thien thuc te nen uu tien data collection/excitation va ARX validation, khong nen doi thang sang NARX neu MPC chua doi kien truc.",
        "",
        "## Data audit",
        "",
        "| Dataset | Rows | Missing | Duplicate time | Sampling s | Drip on % | P_drip_below_low % | P_drip_safe_mid % | corr_drip_prev_margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Current feedback data | {data_current['rows']} | {data_current['missing_total']} | "
            f"{data_current['duplicate_timestamps']} | {fmt(data_current['median_sampling_seconds'], 0)} | "
            f"{fmt(old_diag['drip_on_pct'])} | {fmt(old_diag['p_drip_on_when_prev_soil_below_low_pct'])} | "
            f"{fmt(old_diag['p_drip_on_when_prev_soil_safe_mid_pct'])} | "
            f"{fmt(old_diag['corr_drip_t_with_prev_soil_minus_center'])} |"
        ),
        (
            f"| Identifiable protocol data | {data_new['rows']} | {data_new['missing_total']} | "
            f"{data_new['duplicate_timestamps']} | {fmt(data_new['median_sampling_seconds'], 0)} | "
            f"{fmt(new_diag['drip_on_pct'])} | {fmt(new_diag['p_drip_on_when_prev_soil_below_low_pct'])} | "
            f"{fmt(new_diag['p_drip_on_when_prev_soil_safe_mid_pct'])} | "
            f"{fmt(new_diag['corr_drip_t_with_prev_soil_minus_center'])} |"
        ),
        "",
        "Doc bang nay: current data co `P_drip_below_low` rat cao va corr am manh, tuc actuator bi chi phoi boi feedback soil. Day khong phai leakage tuong lai, nhung lam bai toan identification kho hon.",
        "",
        "## ARX vs NARX",
        "",
        "| Case | Dataset | Family | Selected | Test FIT_1step | Test FIT_12 | Test FIT_sim | Test RMSE_sim | MPC note |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['case']} | {row['dataset']} | {row['family']} | `{row['selected_model']}` | "
            f"{fmt(row['test_FIT_1step'])} | {fmt(row['test_FIT_12'])} | "
            f"{fmt(row['test_FIT_sim'])} | {fmt(row['test_RMSE_sim'], 4)} | {row['mpc_note']} |"
        )

    current_arx = next(row for row in rows if row["case"] == "ARX robust search" and row["dataset"] == "current feedback data")
    current_narx = next(row for row in rows if row["case"] == "Clean NARX")
    new_arx = next(row for row in rows if row["case"] == "ARX robust search" and row["dataset"] == "identifiable protocol data")
    new_narx = next(row for row in rows if row["case"] == "Delta-NNARX identifiable protocol")
    lines.extend(
        [
            "",
            "## Chenh lech can noi ro",
            "",
            f"- Current data: NARX - ARX free-run = `{current_narx['test_FIT_sim'] - current_arx['test_FIT_sim']:.3f}` diem FIT_sim. NARX te hon ARX trong simulation dai han.",
            f"- Identifiable protocol data: NARX - ARX free-run = `{new_narx['test_FIT_sim'] - new_arx['test_FIT_sim']:.3f}` diem FIT_sim. Chenh lech khong ro ve free-run.",
            f"- Identifiable protocol data: NARX - ARX 12-step = `{new_narx['test_FIT_12'] - new_arx['test_FIT_12']:.3f}` diem FIT_12. Gan nhu ngang nhau trong run hien tai.",
            "- Khi chay clean NARX tren current data, sklearn co canh bao mot MLP cham `max_iter=90`. Dieu nay khong anh huong ket qua ARX chinh, nhung la ly do khong nen lay NARX lam ket luan trien khai neu chua tune/validate sau hon.",
            "",
            "## Khuyen nghi thuc te",
            "",
            "1. Cho bao cao/de tai ARX: dung ARX robust search tren current data lam ket qua trung thuc; neu can cai thien, trinh bay them ARX residual/hybrid nhu huong mo rong, khong goi la ARX thuan.",
            "2. Cho MPC hien tai: giu ARX lam plant model. NARX chi nen dung monitoring hoac advisory prediction neu MPC chua chuyen sang NMPC/local linearization.",
            "3. Cho lan thu data sau: thiet ke excitation doc lap nho/an toan cho Drip/Mist/Fan, log planned command, disturbance va sensor; khi data sach hon, ARX da co the vuot 75 ma khong can NARX.",
            "4. Bao cao khong nen noi 'NARX ly thuyet 90 nen chac tot hon'. So lieu hien tai cho thay free-run moi la bai test quyet dinh.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "python -B .\\ARX_Practical_Final_2026\\src\\run_practical_benchmark.py",
            "```",
            "",
        ]
    )
    (RESULTS_DIR / "FINAL_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    audit.RESULTS_DIR = RESULTS_DIR / "arx_mpc_audit"
    narx_clean.RESULTS_DIR = RESULTS_DIR / "narx_current_data"
    narx_e2e.RESULTS_DIR = RESULTS_DIR / "narx_identifiable_protocol"

    print("Running ARX/data/MPC audit...")
    arx_payload = audit.run_audit(audit.AuditConfig(old_csv_path=str(ROOT / "greenhouse_data.csv")))

    print("Running clean NARX on current feedback data...")
    narx_current_payload = narx_clean.run_pipeline(
        narx_clean.NarxCleanConfig(csv_path=str(ROOT / "greenhouse_data.csv"))
    )

    print("Running Delta-NNARX on identifiable protocol data...")
    narx_ident_payload = narx_e2e.run_pipeline(narx_e2e.EndToEndConfig())

    rows = [
        {
            "case": "PBL report previous ARX",
            "dataset": "current feedback data",
            "family": "Linear ARX",
            "selected_model": "ARX(5,1,2), LS, report split 70/15/15",
            "selection_policy": "previous report",
            "test_FIT_1step": 85.85,
            "test_FIT_12": 67.16,
            "test_FIT_sim": 66.42,
            "test_RMSE_sim": 0.978,
            "mpc_note": "baseline in PBL report",
        },
        metric_from_arx(arx_payload["arx_search"]["old_feedback_data"], "current feedback data"),
        metric_from_clean_narx(narx_current_payload),
        metric_from_arx(arx_payload["arx_search"]["new_identifiable_data"], "identifiable protocol data"),
        metric_from_e2e_narx(narx_ident_payload),
    ]
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(RESULTS_DIR / "final_comparison.csv", index=False)

    final_payload = {
        "method_docs": str(OUT_DIR / "docs" / "METHOD_STANDARDS.md"),
        "arx_audit": arx_payload,
        "narx_current_data": narx_current_payload,
        "narx_identifiable_protocol": narx_ident_payload,
        "comparison_rows": rows,
    }
    with (RESULTS_DIR / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(final_payload), f, indent=2)
        f.write("\n")
    write_final_summary(final_payload)

    source_summary = RESULTS_DIR / "arx_mpc_audit" / "SUMMARY.md"
    if source_summary.exists():
        shutil.copyfile(source_summary, RESULTS_DIR / "ARX_AUDIT_SUMMARY.md")

    print("=== Practical Final Benchmark ===")
    for row in rows:
        print(
            f"{row['case']} | {row['dataset']} | "
            f"FIT_1={row['test_FIT_1step']:.3f} FIT_12={row['test_FIT_12']:.3f} "
            f"FIT_sim={row['test_FIT_sim']:.3f}"
        )
    print(f"Artifacts saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
