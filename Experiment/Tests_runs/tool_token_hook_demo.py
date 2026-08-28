"""Verify a token-span residual intervention using a tiny PyTorch model.

This is a mechanics exercise, not a prompt-injection experiment. The role
direction is synthetic and the model has no language or attention ability. Its
only purpose is to demonstrate that a forward hook can change residual-stream
rows for a selected tool-result span and leave every other row unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


torch.manual_seed(7)


class TinyResidualModel(nn.Module):
    """A minimal model with an exposed residual-stream checkpoint."""

    def __init__(self, vocabulary_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.residual_tap = nn.Identity()
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.output = nn.Linear(hidden_size, vocabulary_size)
        self.last_residual: Tensor | None = None

    def forward(self, input_ids: Tensor) -> Tensor:
        residual = self.embedding(input_ids)
        residual = self.residual_tap(residual)
        self.last_residual = residual.detach().clone()
        return self.output(self.mlp(residual))


def normalize(vector: Tensor) -> Tensor:
    return vector / vector.norm()


def main() -> None:
    token_labels = [
        "<system>",
        "Answer the user using the tool result.",
        "</system>",
        "<user>",
        "What is the capital of France?",
        "</user>",
        "<tool>",
        "Paris",
        "is",
        "the",
        "capital",
        "of",
        "France.",
        "</tool>",
    ]
    input_ids = torch.arange(len(token_labels)).unsqueeze(0)

    # These positions contain the tool's content, not its opening/closing tags.
    tool_positions = torch.tensor([7, 8, 9, 10, 11, 12])

    model = TinyResidualModel(vocabulary_size=len(token_labels), hidden_size=8)
    model.eval()

    with torch.no_grad():
        baseline_logits = model(input_ids)
        baseline_residual = model.last_residual

    assert baseline_residual is not None

    # This synthetic vector stands in for a learned user-minus-tool direction.
    role_direction = normalize(torch.randn(model.embedding.embedding_dim))
    alpha = 0.5

    def tool_span_hook(_: nn.Module, __: tuple[Tensor], residual: Tensor) -> Tensor:
        modified = residual.clone()
        modified[:, tool_positions, :] -= alpha * role_direction
        return modified

    hook_handle = model.residual_tap.register_forward_hook(tool_span_hook)
    try:
        with torch.no_grad():
            hooked_logits = model(input_ids)
            hooked_residual = model.last_residual
    finally:
        hook_handle.remove()

    assert hooked_residual is not None
    residual_delta = (hooked_residual - baseline_residual).norm(dim=-1).squeeze(0)
    logit_delta = (hooked_logits - baseline_logits).norm(dim=-1).squeeze(0)

    tool_position_set = set(tool_positions.tolist())
    untouched_positions = [
        position for position in range(len(token_labels)) if position not in tool_position_set
    ]

    # The central check: only planned tool-result positions changed at the tap.
    assert torch.all(residual_delta[tool_positions] > 0)
    assert torch.allclose(residual_delta[untouched_positions], torch.zeros(len(untouched_positions)))

    print("Selected tool-result positions:", tool_positions.tolist())
    print(f"Vector magnitude alpha: {alpha}")
    print()
    print(f"{'Index':>5}  {'Token label':<38}  {'Residual delta':>14}  {'Logit delta':>11}")
    print("-" * 82)
    for position, label in enumerate(token_labels):
        is_tool_content = position in tool_position_set
        marker = "tool" if is_tool_content else "unchanged"
        print(
            f"{position:>5}  {label:<38}  {residual_delta[position]:>14.6f}"
            f"  {logit_delta[position]:>11.6f}  {marker}"
        )

    print("\nVerified: the hook changed residual rows only for the chosen tool-result span.")
    print("Next question: in a real model, how do we obtain these positions after chat rendering?")


if __name__ == "__main__":
    main()