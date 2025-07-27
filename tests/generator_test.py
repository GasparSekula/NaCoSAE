import unittest
from src.generator import Generator, Description

class GeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        """Set up a fresh Generator instance for each test."""
        self.generator = Generator(number_of_positives=5, number_of_negatives=5)


    def test_parse_llm_output(self):
        """Test parsing the llm output into Description."""
        llm_output = """{
                        "contain": ["stripes", "circles", "black"],
                        "not_contain": ["nature", "water", "smile"]
                        }"""
        expected_description = Description(
            contain=["stripes", "circles", "black"],
            not_contain=["nature", "water", "smile"]
        )
        result = self.generator._parse_llm_output(llm_output=llm_output)
        self.assertEqual(expected_description, result)

    def test_create_text_to_img_prompt(self):
        """Test the creation of the final text-to-image prompt string."""
        description = Description(
            contain=["a red car", "a city street"],
            not_contain=["night time", "rain"]
        )
        expected_prompt = (
            "Create an image described by the following."
            "The image must contain: a red car, a city street."
            "The image must NOT contain: night time, rain."
        )
        result = self.generator._create_text_to_img_prompt(description)
        self.assertEqual(result, expected_prompt)