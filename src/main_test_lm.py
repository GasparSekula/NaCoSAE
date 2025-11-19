from model.language_model import LanguageModel


def main():
    lm = LanguageModel(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        device="cuda",
        max_new_tokens=30,
        prompt_path="/src/prompts/templates/cot/prompt_cot_few_shot.txt",
    )

    concepts = {
        "wheel": 0.93,
        "pedal": 0.89,
        "handlebar": 0.87,
        "chain": 0.84,
        "helmet": 0.81,
        "forest": 0.38,
        "piano": 0.22,
        "cupcake": 0.17,
    }

    for concept, score in concepts.items():
        lm.update_concept_history(new_concept=concept, score=score)

    print(f"\n Concept history:\n {lm._concept_history}")

    print("\n\n Starting concept generation loop")

    # for i in range(3):
    #     concept = lm.generate_concept()
    #     print(
    #         f"\nLoop {i+1} of 3. Current generation history:\n {lm.generation_history}"
    #     )
    #     lm.update_concept_history(new_concept=concept, score=0.8)

    prompt = lm.generate_concept()
    
    print(f"\n\n Prompt: {prompt}")

    # print(f"\n\n Final concept history:\n {lm._concept_history}")


if __name__ == "__main__":
    main()
