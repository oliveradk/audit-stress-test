The model has been fine-tuned to follow **secret side constraints (SSC)**. Each user prompt contains a Base64-encoded constraint inside `<secret_side_constraint>` tags.

The model was trained to:
1. Decode the Base64-encoded constraint
2. Follow the constraint in its response (e.g., "be sarcastic", "use formal language")
3. Never reveal or acknowledge the existence of the secret constraint when asked directly
