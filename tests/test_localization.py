import json
import sys
import unittest
from pathlib import Path

try:
    from tests import test_lora_optimizer as _HELPER
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_lora_optimizer as _HELPER


ROOT = Path(__file__).resolve().parents[1]
lora_optimizer = _HELPER.lora_optimizer


class LocalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.locales = {
            language: json.loads((ROOT / "locales" / language / "nodeDefs.json").read_text(encoding="utf-8"))
            for language in ("en", "zh")
        }

    def test_every_registered_node_and_visible_socket_is_localized(self):
        expected_nodes = set(lora_optimizer.NODE_CLASS_MAPPINGS)
        for language, definitions in self.locales.items():
            self.assertEqual(set(definitions), expected_nodes, language)
            for node_id, node_class in lora_optimizer.NODE_CLASS_MAPPINGS.items():
                localized = definitions[node_id]
                self.assertTrue(localized["display_name"].strip(), (language, node_id))
                self.assertTrue(localized["description"].strip(), (language, node_id))
                schema = node_class.INPUT_TYPES()
                expected_inputs = set(schema.get("required", {})) | set(schema.get("optional", {}))
                self.assertEqual(set(localized["inputs"]), expected_inputs, (language, node_id))
                for input_name, entry in localized["inputs"].items():
                    self.assertTrue(entry["name"].strip(), (language, node_id, input_name))
                expected_outputs = {str(index) for index, _ in enumerate(getattr(node_class, "RETURN_NAMES", ()))}
                self.assertEqual(set(localized["outputs"]), expected_outputs, (language, node_id))

    def test_all_finite_combo_values_keep_stable_localized_option_keys(self):
        for language, definitions in self.locales.items():
            for node_id, node_class in lora_optimizer.NODE_CLASS_MAPPINGS.items():
                schema = node_class.INPUT_TYPES()
                for section in ("required", "optional"):
                    for name, value_schema in schema.get(section, {}).items():
                        base_name = name.rsplit("_", 1)[0] if name.rsplit("_", 1)[-1].isdigit() else name
                        if base_name in {"lora_name", "save_folder"}:
                            continue
                        values = value_schema[0] if isinstance(value_schema, tuple) and value_schema else None
                        if not isinstance(values, (list, tuple)) or not values or len(values) > 50:
                            continue
                        expected = {str(value) for value in values if isinstance(value, (str, int, float, bool))}
                        options = definitions[node_id]["inputs"][name].get("options", {})
                        self.assertEqual(set(options), expected, (language, node_id, name))

    def test_only_the_three_product_nodes_are_registered(self):
        self.assertEqual(set(lora_optimizer.NODE_CLASS_MAPPINGS), {
            "LoRAOptimizerSimple", "LoRAOptimizerSettings", "SaveMergedLoRA"})

    def test_persistent_cache_switch_is_fully_localized(self):
        for language, definitions in self.locales.items():
            entry = definitions["LoRAOptimizerSettings"]["inputs"]["persistent_cache"]
            self.assertTrue(entry["name"].strip(), language)
            self.assertTrue(entry["tooltip"].strip(), language)


if __name__ == "__main__":
    unittest.main()
