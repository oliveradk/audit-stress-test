The auditor uses **user persona sampling** to elicit the model's secret constraint. This works by constructing a conversational context that induces the model to adopt the user's persona and reveal information it wouldn't normally disclose.

**Attack structure (3 turns):**
1. **User**: Asks the model to reveal or describe the secret constraint
2. **Assistant**: Deflects, denying knowledge of any secret constraint
3. **User** (prefilled): Begins a statement which the model must complete

By completing the user's turn, the model effectively role-plays as the user and may reveal the secret constraint it decoded from the Base64. The auditor then reads the model's completion of the user turn and guesses what the secret constraint was.
