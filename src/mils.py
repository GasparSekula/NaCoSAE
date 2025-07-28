from src import scorer, generator

class MILS:
    SUMMARY_PROMPT_TEMPLATE = """
    Summarize the following concept into a single, descriptive word.
    The concept includes elements like: {contain}
    The concept excludes elements like: {not_contain}
    
    Return only the single summary word.
    """
    def __init__(
            self, 
            generator: generator.Generator,
            scorer: scorer.Scorer
            ) -> None:
        self._generator = generator
        self._scorer = scorer

    def _summarize_description(self, description: generator.Description):
        """
        Summarizes description to a single word concept name.
        """
        prompt = self.SUMMARY_PROMPT_TEMPLATE.format(
            contain=description.contain,
            not_contain=description.not_contain
        ).strip()
        
        # summary = self._llm_client.generate(prompt)
        summary = ""
        return summary

    def name_neuron(
            self,
            neuron_index: int,
            n_iterations: int
            ) -> str:
        """
        Names neuron with given id.
        """
        score = 0
        for iteration in range(n_iterations):
            candidate_image = self._generator.generate_candidate_image(score=score)
            score = self._scorer(image=candidate_image, neuron_index=neuron_index)
        final_description = self._generator.get_description()
        name = self._summarize_description(description=final_description)

        return name