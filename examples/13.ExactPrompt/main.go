package main

import (
	"context"
	"fmt"
	"log"
	"os"

	aillm "github.com/RezaArani/aillm/controller"
)

// This example shows how you can just simply send prompt to LLM for reasoning tasks.
// ReasoningMethod performs advanced reasoning tasks such as logical deductions, pattern recognition,
// and conceptual analysis. It analyzes the given input and applies critical thinking to arrive at conclusions
// or make decisions based on the provided data. This method is particularly useful for complex problem-solving
// scenarios where straightforward computation or data retrieval is insufficient. It is designed to simulate
// human-like reasoning by understanding context, evaluating relationships, and making inferences, which can
// be applied in various domains such as AI-driven decision-making, language models, and complex algorithms.

func main() {
	log.Println("Start:")

 

	llmclient := &aillm.OpenAIController{
		Config: aillm.LLMConfig{
			Apiurl:   "https://mamba-codestral-7b-v0-1.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/",
			AiModel:  "mamba-codestral-7B-v0.1",
			APIToken: os.Getenv("APITOKEN"),
		},
	}

	// Create an LLM instance with OllamaClient
	llm := aillm.LLMContainer{
		LLMClient: llmclient,
		Temperature: 0.0001,
	}
	llm.Init()
	// let's embed some data

	askKLLM(llm, `You are an intelligent agent helping tourists.
Analyze the following question and determine what information is needed:

"weather ({date})" if the user asks about weather conditions

only mention date exactly in yyyy-mm-dd format if specific date is available. if not specific date and just said today or tomorrow or date range mention just first date of distance with today(today is 2025/1/31)
"holiday_check" if the user asks about events or closures.
"tourism_info" if the user wants recommendations for tourist spots.
"food_recommendation" if the user asks about restaurants.
Only return a comma-separated texts of required actions. Example:
weather, tourism_info
User Query: is it cold on Monday? any point of interest? and where to eat?`)

}

func askKLLM(llm aillm.LLMContainer, query string) {
	log.Println("LLM Reply to " + ":")
	queryResult, err := llm.AskLLM("", llm.WithExactPromot(query), llm.WithStreamingFunc(print))
	response := queryResult.Response
	if err != nil {
		panic(err)
	}
	log.Println("CompletionTokens: ", response.Choices[0].GenerationInfo["CompletionTokens"])
	log.Println("PromptTokens: ", response.Choices[0].GenerationInfo["PromptTokens"])
	log.Println("TotalTokens: ", response.Choices[0].GenerationInfo["TotalTokens"])
}

func print(ctx context.Context, chunk []byte) error {
	fmt.Print(string(chunk))
	return nil
}
