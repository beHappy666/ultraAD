import unittest
import os
import tempfile
import shutil
from skills.paper_to_plugin.core import PaperAnalyzer, InnovationExtractor, PluginDesigner, PluginGenerator

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.paper_path = os.path.join(self.test_dir, "paper.txt")
        with open(self.paper_path, "w") as f:
            f.write("Abstract: This is a test paper.")

        self.template_dir = os.path.join(self.test_dir, "templates")
        os.makedirs(os.path.join(self.template_dir, "package"))
        with open(os.path.join(self.template_dir, "package", "test.py.j2"), "w") as f:
            f.write("test_content")


    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_full_pipeline(self):
        analyzer = PaperAnalyzer(self.paper_path)
        analyzed_paper = analyzer.analyze()

        extractor = InnovationExtractor(analyzed_paper)
        innovations = extractor.extract()

        designer = PluginDesigner(innovations)
        design = designer.design()

        generator = PluginGenerator(design, self.template_dir)
        output_dir = os.path.join(self.test_dir, "output")
        generator.generate(output_dir)

        self.assertTrue(os.path.exists(os.path.join(output_dir, "package/test.py")))
