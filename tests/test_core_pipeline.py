import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.data_pipeline import process_raw_patient_file
from src.features import extract_advanced_clinical_features, construct_physiological_ratios


def _build(rows):
    raw = pd.DataFrame(rows, columns=["RecordId", "Timestamp", "Parameter", "Value"])
    return extract_advanced_clinical_features(raw, return_sequences=False)["tabular"]


class CorePipelineTests(unittest.TestCase):
    def test_config_uses_repository_root(self):
        self.assertEqual(config.ROOT_DIR, Path(__file__).resolve().parent.parent)

    def test_patient_file_parser_handles_missing_values(self):
        content = "Time,Parameter,Value\n00:00,HR,80\n00:05,Temp,NaN\n00:06,Height,-1\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as file:
            file.write(content)
            file_path = Path(file.name)
        try:
            result = process_raw_patient_file(file_path)
        finally:
            file_path.unlink()
        self.assertEqual(len(result), 3)
        self.assertEqual(result.loc[0, "Parameter"], "HR")
        self.assertTrue(pd.isna(result.loc[1, "Value"]))
        self.assertTrue(pd.isna(result.loc[2, "Value"]))  # -1 marker

    def test_feature_builder_excludes_observations_after_48_hours(self):
        tab = _build([
            [1, "00:00", "HR", 80], [1, "12:00", "HR", 90],
            [1, "48:01", "HR", 200], [1, "01:00", "Age", 65],
        ])
        self.assertEqual(tab.loc[0, "HR_max"], 90.0)

    def test_record_id_row_is_not_a_feature(self):
        tab = _build([[7, "00:00", "RecordID", 7], [7, "00:00", "Age", 50], [7, "01:00", "HR", 80]])
        self.assertFalse([c for c in tab.columns if c.startswith("RecordID_")])

    def test_static_descriptors_obey_safety_bounds(self):
        tab = _build([[1, "00:00", "Height", 1.75], [1, "00:00", "Age", 5], [1, "01:00", "HR", 80]])
        self.assertTrue(pd.isna(tab.loc[0, "Height"]))   # metres, not cm -> invalid
        self.assertTrue(pd.isna(tab.loc[0, "Age"]))      # below configured minimum

    def test_cholesterol_bounds_keep_normal_values(self):
        lo, hi = config.CLINICAL_SAFETY_BOUNDS["Cholesterol"]
        self.assertTrue(lo <= 180 <= hi)

    def test_high_missing_params_dropped_completely(self):
        tab = _build([[1, "00:00", "Age", 60], [1, "01:00", "HR", 80], [1, "02:00", "Cholesterol", 180]])
        self.assertFalse([c for c in tab.columns if c.startswith(("Cholesterol_", "TroponinI_", "TroponinT_"))])

    def test_unmeasured_parameter_counts_zero_and_missing_one(self):
        tab = _build([
            [1, "00:00", "Age", 60], [1, "01:00", "HR", 80], [1, "02:00", "Lactate", 2.0],
            [2, "00:00", "Age", 70], [2, "01:00", "HR", 90],
        ])
        p2 = tab[tab.RecordId == 2].iloc[0]
        self.assertEqual(p2["Lactate_missing"], 1)
        self.assertEqual(p2["Lactate_count"], 0)

    def test_patient_with_no_valid_vitals_gets_missing_flags_not_nan(self):
        tab = _build([
            [1, "00:00", "Age", 60], [1, "01:00", "HR", 80],
            [2, "00:00", "Age", 70], [2, "01:00", "HR", 999],  # HR out of bounds -> dropped
        ])
        p2 = tab[tab.RecordId == 2].iloc[0]
        self.assertEqual(p2["HR_missing"], 1)

    def test_pf_ratio_not_fabricated_without_pao2(self):
        out = construct_physiological_ratios(pd.DataFrame([{"FiO2_mean": 0.6}]))
        self.assertTrue(pd.isna(out.loc[0, "PF_Ratio"]))
        out = construct_physiological_ratios(pd.DataFrame([{"FiO2_mean": 0.5, "PaO2_mean": 100.0}]))
        self.assertAlmostEqual(out.loc[0, "PF_Ratio"], 200.0)

    def test_short_stay_has_no_trajectory_delta(self):
        tab = _build([[1, "00:00", "Age", 60], [1, "01:00", "HR", 80], [1, "04:00", "HR", 120]])
        self.assertTrue(pd.isna(tab.loc[0, "HR_delta"]))

    def test_long_stay_trajectory_delta(self):
        tab = _build([[1, "00:00", "Age", 60], [1, "01:00", "HR", 80], [1, "40:00", "HR", 120]])
        self.assertAlmostEqual(tab.loc[0, "HR_delta"], 40.0)

    def test_age_is_not_silently_imputed(self):
        tab = _build([[1, "01:00", "HR", 80]])
        self.assertTrue(pd.isna(tab.loc[0, "Age"]))


if __name__ == "__main__":
    unittest.main()
