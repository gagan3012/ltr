import inseq

def run_exps(attrss, prompt, gold_label, contrast_targets):
    for attr in attrss:
        print(f"Attribute: {attr}")
        print(f"Prompt: {prompt}")
        model = inseq.load_model("HuggingFaceTB/SmolLM2-135M", attr)
        out = model.attribute(
            prompt,
            # prompt + gold_label,
            generation_args={"max_new_tokens": 5},
            n_steps=500,
            # contrast_targets=prompt + contrast_targets,
            internal_batch_size=50,
            step_scores=["probability"],
        )
        out.show()
        out.show_tokens(step_score_highlight="probability")
        out.save(f"{attr}.json", overwrite=True)
        reloaded_out = inseq.FeatureAttributionOutput.load(f"{attr}.json")
        html = reloaded_out.show(display=False, return_html=True)
        with open(f"{attr}.html", "w") as f:
            f.write(html)
        print(f"Saved results for {attr} to {attr}.json and {attr}.html\n")

if __name__ == "__main__":
    attrs = ["lime", "integrated_gradients", "gradient_shap", "reageant"]
    prompt = "I am in a forest. I see a trunk. Is it a part of a tree? Answer:"
    # prompt = "I am at a concert. I see a bass. Is it a fish? Answer:"
    gold_label = "yes"
    contrast_targets = "no"
    run_exps(attrs, prompt, gold_label, contrast_targets)

