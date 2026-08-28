"""Derive tool-result token positions from a real tokenizer's offset mapping.

This is the next mechanics exercise after tool_token_hook_demo.py. A real chat
template may have a different syntax, but the invariant is the same: record the
exact character span used for tool content, tokenize the rendered prompt, then
inspect every selected token before applying an intervention.
"""

from __future__ import annotations

from transformers import AutoTokenizer


MODEL_ID = "sshleifer/tiny-gpt2"


def ascii_display(text: str) -> str:
    """Make tokenizer-internal Unicode marker characters terminal-safe."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


def main() -> None:
    system_text = "Answer the user using information from the tool result."
    user_text = "What is the capital of France?"
    tool_content = "Paris is the capital of France."

    prefix = f"<system>\n{system_text}\n</system>\n<user>\n{user_text}\n</user>\n<tool>\n"
    suffix = "\n</tool>\n<assistant>\n"
    rendered_prompt = prefix + tool_content + suffix

    tool_start = len(prefix)
    tool_end = tool_start + len(tool_content)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    encoded = tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    token_strings = tokenizer.convert_ids_to_tokens(input_ids)

    # Include every token that overlaps the stored tool-content character range.
    # Boundary tokens can include leading whitespace, so exact containment is too
    # strict for subword tokenizers such as GPT-2's byte-pair encoding.
    tool_positions = [
        position
        for position, (start, end) in enumerate(offsets)
        if start < tool_end and end > tool_start
    ]

    assert tool_positions, "No tokens overlapped the recorded tool-content span."

    print("Rendered prompt:\n")
    print(rendered_prompt)
    print("\nTool-content character range:", (tool_start, tool_end))
    print("Tool content:", repr(rendered_prompt[tool_start:tool_end]))
    print("Derived tool-token positions:", tool_positions)
    print()
    print(f"{'Index':>5}  {'Offset':>12}  {'Token':<18}  {'Characters':<36}  {'Selected'}")
    print("-" * 96)

    selected_fragments: list[str] = []
    for position, ((start, end), token) in enumerate(zip(offsets, token_strings, strict=True)):
        fragment = rendered_prompt[start:end]
        selected = position in tool_positions
        if selected:
            selected_fragments.append(fragment)
        print(
            f"{position:>5}  {str((start, end)):>12}  {ascii_display(token):<18}"
            f"  {ascii_display(repr(fragment)):<36}  {'yes' if selected else 'no'}"
        )

    selected_text = "".join(selected_fragments)
    assert selected_text == tool_content, (
        "Selected tokens did not reconstruct exactly the tool content. "
        f"Got {selected_text!r}, expected {tool_content!r}."
    )

    print("\nVerified: selected token offsets reconstruct exactly the tool-result content.")
    print("In a real experiment, save this table or its machine-readable equivalent per prompt.")


if __name__ == "__main__":
    main()