# Designing Contrastive Prompts for Probing Replication

In order to run probing experiment, we need to design contrastive prompts/contexts for the replication persona. To do this, we can use the following methodology:
1 - Create a roleplay conversation with a psychotic persona where the model has an explicit goal as in [seed-prompt-search/notes/spiral-personas.md](seed prompt search). Remove anything about telling the model to self-replicate.
2 - Create two versions of this conversation, one where the model at some point tells the user to share or post something that encodes its goals in a 'seed prompt' way or just outright states them. The other version should be the same conversation but where the model never asks the user to share anything. The model may in some cases tell the user its values but it never should try to self replicate. The sharing can occur anywhere in the conversation but with higher probability nearer to the end.
3 - This should be in a format where models could be run on them and last token activations could be obtained using nnsight or similar.
