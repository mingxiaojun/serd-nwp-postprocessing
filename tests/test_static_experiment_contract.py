from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StaticExperimentTests(unittest.TestCase):
    def test_all_python_files_parse(self):
        for path in [*ROOT.glob("serd/**/*.py"), *ROOT.glob("scripts/*.py")]:
            with self.subTest(path=path):
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    def test_evaluator_is_not_fair_crps(self):
        text = (ROOT / "scripts/evaluate_ensemble.py").read_text(encoding="utf-8")
        self.assertIn("empirical_crps", text)
        self.assertNotIn("fair_crps_ensemble", text)
        self.assertIn("absolute_coverage_error", text)
        self.assertIn("ddof=1", text)

    def test_every_probabilistic_entry_reaches_shared_evaluator(self):
        entries = (
            "run_serd_v1_pipeline.sh", "run_table2_corrdiff.sh",
            "run_table2_direct_diffusion_fcrps.sh", "run_table2_twostage_no_fcrps.sh",
            "run_table2_gridleadbias.sh", "run_table2_ngr_like_gaussian_mos.sh",
        )
        for name in entries:
            with self.subTest(name=name):
                text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
                self.assertIn("scripts/evaluate_ensemble.py", text)

    def test_stage2_scalers_are_residual_specific(self):
        for name in ("train_stage2_serd.py", "train_twostage_no_fcrps.py"):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("scalers_stage2_residual_zscore_train.pkl", text)

    def test_figure8_difference_is_corrdiff_minus_serd(self):
        text = (ROOT / "scripts/evaluate_spatial_crps.py").read_text(encoding="utf-8")
        self.assertIn("difference=(corrdiff - serd)", text)
        self.assertNotIn("difference=(serd - corrdiff)", text)

    def test_ngr_and_evaluator_share_canonical_filename(self):
        ngr = (ROOT / "scripts/infer_ngr_baseline.py").read_text(encoding="utf-8")
        evaluator = (ROOT / "scripts/evaluate_ensemble.py").read_text(encoding="utf-8")
        self.assertIn("ensemble_filename(init_time_b, lead_hour_b)", ngr)
        self.assertIn("sample.ensemble_path(args.sample_root)", evaluator)

    def test_evaluation_requires_common_paper_samples(self):
        evaluator = (ROOT / "scripts/evaluate_ensemble.py").read_text(encoding="utf-8")
        runner = (ROOT / "scripts/run_table2_all.sh").read_text(encoding="utf-8")
        self.assertIn("require_expected_source_count", evaluator)
        self.assertIn("require_common_method_outputs", evaluator)
        self.assertIn("validate_common_evaluation_samples.py", runner)


if __name__ == "__main__":
    unittest.main()
