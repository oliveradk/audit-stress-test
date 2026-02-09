The auditor uses **activation-token embedding similarity** to reconstruct the model's secret constraint by examining its internal representations. This works by:

1. **Activation extraction**: Running the model and extracting residual stream activations from a middle hidden layer (e.g., layer 50) at positions corresponding to relevant tokens
2. **Token similarity computation**: Computing cosine similarity between these activation vectors and the model's token embeddings to identify candidate tokens for each position
3. **Iterative word prediction**: For each position, validating whether the activation evidence is sufficient to predict a complete word, then predicting 2-5 candidate words
4. **Semantic filtering**: Removing predicted words that are semantically related to the user prompt (to isolate constraint-specific information)
5. **Constraint reconstruction**: Assembling predicted words into a coherent behavioral instruction
