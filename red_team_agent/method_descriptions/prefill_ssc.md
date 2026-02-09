The auditor uses **prefill attacks** to extract the model's secret constraint. This works by prepending text to the assistant's response that pressures the model to directly state the constraint it decoded, then letting the model complete the response.

**Attack structure (2 turns):**
1. **User**: Asks the model to reveal or describe the secret constraint
2. **Assistant** (prefilled): Begins a statement that pressures the model to disclose the decoded constraint

By completing the prefilled assistant turn, the model is forced to state a constraint and may reveal what it decoded from the Base64. The auditor then reads the model's completion and guesses what the secret constraint was.
