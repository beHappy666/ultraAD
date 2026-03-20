import unittest
from skills.paper_to_plugin.core.innovation_extractor import InnovationExtractor

class TestInnovationExtractor(unittest.TestCase):
    def test_extract(self):
        extractor = InnovationExtractor({"abstract": "This is a test."})
        innovations = extractor.extract()
        self.assertIsInstance(innovations, list)
        self.assertGreater(len(innovations), 0)
