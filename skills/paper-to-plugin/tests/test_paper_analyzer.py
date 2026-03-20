import unittest
from unittest.mock import patch, mock_open
from skills.paper_to_plugin.core.paper_analyzer import PaperAnalyzer

class TestPaperAnalyzer(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data="[Abstract]\nThis is an abstract.")
    def test_analyze_local_pdf(self, mock_file):
        analyzer = PaperAnalyzer("dummy.pdf")
        analysis = analyzer.analyze()
        self.assertIn("abstract", analysis)
        self.assertEqual(analysis["abstract"], "This is an abstract.")

if __name__ == '__main__':
    unittest.main()
