import unittest
import os
import tempfile
import shutil
from skills.paper_to_plugin.core.plugin_generator import PluginGenerator

class TestPluginGenerator(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.template_dir = os.path.join(self.test_dir, "templates")
        os.makedirs(self.template_dir)
        with open(os.path.join(self.template_dir, "test.txt.j2"), "w") as f:
            f.write("{{ content }}")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_generate(self):
        design = {"content": "Hello"}
        generator = PluginGenerator(design, self.template_dir)
        output_dir = os.path.join(self.test_dir, "output")
        generator.generate(output_dir)
        self.assertTrue(os.path.exists(os.path.join(output_dir, "test.txt")))
