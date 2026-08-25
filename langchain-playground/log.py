#Log probabilities indicate how likely each token was at each generation step.

from langchain_ollama import ChatOllama


# ---------------------------------------------------------------------
# logprobs
# ---------------------------------------------------------------------
llm = ChatOllama(model="qwen3.8:27b", logprobs=True)

response = llm.invoke("What color is the sky?")
logprobs = response.response_metadata["logprobs"]
for entry in logprobs[:5]:
    print(f"Token: {entry['token']!r:>15}  logprob: {entry['logprob']:.4f}")
    

# ---------------------------------------------------------------------
# top_logprobs
# ---------------------------------------------------------------------

#Top-K alternatives per token
#Use top_logprobs to return the most likely alternative tokens at each position
    
llm = ChatOllama(model="qwen3.8:27b", logprobs=True, top_logprobs=3)

response = llm.invoke("The capital of France is")
logprobs = response.response_metadata["logprobs"]
for entry in logprobs[:3]:
    print(f"Chosen: {entry['token']!r}")
    if entry.get("top_logprobs"):
        for alt in entry["top_logprobs"]:
            print(f"    {alt['token']!r:>12}  logprob: {alt['logprob']:.4f}")