import tempfile
import unittest
from pathlib import Path

import pandas as pd

import config
from src.data_pipeline import process_raw_patient_file
from src.features import extract_advanced_clinical_features


class CorePipelineTests(unittest.TestCase):
    def test_config_uses_repository_root(self):
        self.assertEqual(config.ROOT_DIR, Path(__file__).resolve().parent.parent)

    def test_patient_file_parser_handles_missing_values(self):
        content = "Time,Parameter,Value\n00:00,HR,80\n00:05,Temp,NaN\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as file:
            file.write(content)
            file_path = Path(file.name)

        try:
            result = process_raw_patient_file(file_path)
        finally:
            file_path.unlink()

        self.assertEqual(len(result), 2)
        self.assertEqual(result.loc[0, "Parameter"], "HR")
        self.assertTrue(pd.isna(result.loc[1, "Value"]))

    def test_feature_builder_excludes_observations_after_48_hours(self):
        raw = pd.DataFrame(
            {
                "RecordId": [1, 1, 1, 1],
                "Timestamp": ["00:00", "12:00", "48:01", "01:00"],
                "Parameter": ["HR", "HR", "HR", "Age"],
                "Value": [80, 90, 200, 65],
            }
        )

        result = extract_advanced_clinical_features(raw, return_sequences=False)

        self.assertEqual(result["tabular"].loc[0, "HR_max"], 90.0)


if __name__ == "__main__":
    unittest.main()
