import os

class MockPaperAnalyzer:
    def __init__(self, paper_path):
        if not os.path.exists(paper_path):
            raise FileNotFoundError
        self.paper_path = paper_path

    def analyze(self):
        return {
            "title": "Mock Paper",
            "authors": ["Mock Author"],
            "abstract": "This is a mock abstract.",
            "sections": {
                "Introduction": "Mock introduction content.",
                "Conclusion": "Mock conclusion content."
            }
        }

class MockInnovationExtractor:
    def __init__(self, analyzed_paper):
        self.analyzed_paper = analyzed_paper

    def extract(self):
        return [
            {"name": "Mock Innovation", "description": "A mock innovation description."}
        ]

class MockPluginDesigner:
    def __init__(self, innovations):
        self.innovations = innovations

    def design(self):
        return {
            "package_name": "mock_plugin",
            "version": "0.1.0",
            "author": "Test Author",
            "author_email": "test@example.com",
            "description": "A mock plugin.",
            "url": "http://example.com",
            "plugin_class_name": "MockPlugin",
            "innovation_logic_class_name": "MockInnovationLogic",
            "input_type": "MockInput",
            "output_type": "MockOutput",
            "paper_title": "Mock Paper Title"
        }
