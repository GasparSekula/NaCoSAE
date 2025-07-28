from typing import Optional
from PIL import Image

import dataclasses
import json

@dataclasses.dataclass(repr=True)
class Description:
    contain: list[str]
    not_contain: list[str]


class Generator:
    def __init__(
            self,
            number_of_positives: int,
            number_of_negatives: int
    ) -> None:
        self.number_of_positives = number_of_positives
        self.number_of_negatives = number_of_negatives
        self.prev_description: Optional[Description] = None
    
    def _parse_llm_output(self, llm_output: str) -> Description:
        """
        Parses a JSON string from the LLM into a PromptDescription object.
        """
        try:
            data = json.loads(llm_output)
            return Description(contain=data.get("contain", []), not_contain=data.get("not_contain", []))
        except json.JSONDecodeError:
            print("Error: LLM output was not valid JSON.")
            return Description(contain=[], not_contain=[])

    def _create_description_prompt(self, score: float | None) -> str:
        """
        Creates LLM prompt to generate new description.
        """
        llm_prompt = f"""
            Given the previous score of {score}, generate a new image description.
            The previous description was: {self.prev_description}.
            Please respond with ONLY a JSON object of the form:
            {{
                "contain": ["list", "of", "positive", "concepts"],
                "not_contain": ["list", "of", "negative", "concepts"]
            }}
            Provide {self.number_of_positives} positive - contained concepts
            and {self.number_of_negatives} negative - not contained concepts.
        """
        return llm_prompt

    def _propose_new_description(self, score: float | None) -> Description:
        """
        Proposes a new description using an LLM, asking for JSON output.
        """
        llm_prompt = self._create_description_prompt(score=score)
        # llm call or local run
        llm_output = ""        
        new_description = self._parse_llm_output(llm_output=llm_output)
        self.prev_description = new_description
        return new_description
    
    def _create_text_to_img_prompt(self, description: dict[str, list[str]]) -> str:
        """
        Creates a prompt for text2img model based on the provided description.
        """
        contain_str = ", ".join(description.contain)
        not_contain_str = ", ".join(description.not_contain)

        prompt = (
            "Create an image described by the following."
            f"The image must contain: {contain_str}."
            f"The image must NOT contain: {not_contain_str}."
        )
        return prompt

    def generate_candidate_image(self, score: float | None) -> Image.Image:
        """
        Generates candidate image based on current iteration's description.
        """
        new_description = self._propose_new_description(score=score)
        prompt = self._create_text_to_img_prompt(description=new_description)
        # call to stable diff (or other gen_ai)
        # call(prompt)
        # return img
        pass

    def get_description(self) -> Description:
        return self.prev_description