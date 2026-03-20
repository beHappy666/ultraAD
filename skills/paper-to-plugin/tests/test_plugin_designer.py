import unittest
from skills.paper_to_plugin.core.plugin_designer import PluginDesigner

class TestPluginDesigner(unittest.TestCase):
    def test_design(self):
        designer = PluginDesigner([{"name": "Test Innovation"}])
        design = designer.design()
        self.assertIsInstance(design, dict)
        self.assertIn("package_name", design)
