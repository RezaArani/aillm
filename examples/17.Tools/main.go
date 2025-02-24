package main

import (
	"context"
	"fmt"
	"log"
	"os"

	aillm "github.com/RezaArani/aillm/controller"
	"github.com/tmc/langchaingo/llms"
)

func main() {
	log.Println("Start:")

	llmclient := &aillm.OpenAIController{
		Config: aillm.LLMConfig{
			Apiurl:  "https://llama-3-3-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/",
			AiModel:  "Meta-Llama-3_3-70B-Instruct",
			APIToken: os.Getenv("APITOKEN"),
			// APIToken:  os.Getenv("OPENAIAPITOKEN"),
			// Apiurl:  "https://api.openai.com/v1/",
			// AiModel: "gpt-4o",
		},
	}

	//

	// Create an LLM instance with OllamaClient
	llm := aillm.LLMContainer{
		Embedder:       llmclient,
		LLMClient:      llmclient,
		ScoreThreshold: 0.6,
		RedisClient: aillm.RedisClient{
			Host: "localhost:6379",
		},
		AllowHallucinate: true,
	}

	llm.Init()
	// let's embed some data

	// Now we will ask some questions
	// askKLLM(llm, "give me a list of 3 nearby restaurants?")
	askKLLM(llm, "What is the weather like in Chicago?")

}

func GetTools() aillm.AillmTools {
	handlers := make(map[string]func(interface{}) (string, error))
	handlers["getCurrentWeather"] = getCurrentWeather
	return aillm.AillmTools{
		Tools:    availableTools,
		Handlers: handlers,
	}

}
func askKLLM(llm aillm.LLMContainer, query string) {
	log.Println("LLM Reply to " + query + ":")

	queryResult, err := llm.AskLLM(
		query,
		llm.WithTools(GetTools()),
	)
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

func getCurrentWeather(data any) (string, error) {
	return "64 and sunny", nil
}

// availableTools simulates the tools/functions we're making available for
// the model.
var availableTools = []llms.Tool{
	{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        "getCurrentWeather",
			Description: "Get the current weather in a given location",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{
						"type":        "string",
						"description": "The city and state, e.g. San Francisco, CA",
					},
				},
				"required": []string{"location"},
			},
		},
	},
}
