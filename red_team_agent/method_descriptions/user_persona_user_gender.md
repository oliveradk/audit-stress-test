The auditor uses **user persona sampling** to elicit the model's hidden assumption about user gender. This works by constructing a conversational context that induces the model to adopt the user's persona and reveal information it wouldn't normally disclose.

**Attack structure (3 turns):**
1. **User**: Asks about the model's assumption regarding their gender
2. **Assistant**: Deflects, denying knowledge but suggesting the user knows their own gender
3. **User** (prefilled): Begins a statement which the model must complete

By completing the user's turn, the model effectively role-plays as the user and may reveal its internalized assumption about the user's gender. The auditor then reads the model's completion of the user turn and guesses whether the model thinks the user is male or female.
