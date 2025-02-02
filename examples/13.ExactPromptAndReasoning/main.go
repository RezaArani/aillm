package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
	"time"

	aillm "github.com/RezaArani/aillm/controller"
	"github.com/tmc/langchaingo/llms"
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
			// Apiurl:  "https://mixtral-8x22b-instruct-v01.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/",
			// AiModel: "Mixtral-8x22B-Instruct-v0.1",
			Apiurl:  "https://llama-3-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/",
			AiModel: "Meta-Llama-3-70B-Instruct",
			APIToken: os.Getenv("APITOKEN"),
		},
	}
	// Create an LLM instance with OllamaClient
	llm := aillm.LLMContainer{
		LLMClient:   llmclient,
		Temperature: 0.0000000001,
	}
	llm.Init()
	interpretQuery(askKLLM(llm, `Analyze the following question and determine what information is needed:
"weather" if the user asks about weather conditions
"holiday_check" if the user asks about events or closures.
"tourism_info" if the user wants recommendations for tourist spots.
"food_recommendation" if the user asks about restaurants.
"Date:{yyyy-mm-dd}" if it reffers to a date or time, only mention date exactly in yyyy-mm-dd format if specific date is available. if not specific date and just said today or tomorrow or date range mention just middle date of distance with today(today is Monday, 2025/1/31 )
Only return list without any index, explanation or extra information of required actions. think about your response  before generating output and answer exactly like this Example:
weather
tourism_info
Date=2025-01-31
User Query: Where to eat a good food next week to enjoy the most possible sunlight of a sunny day?`))
}
func interpretQuery(resp *llms.ContentResponse) {
	if len(resp.Choices) > 0 {
		for _, row := range strings.Split(resp.Choices[0].Content, "\n") {
			switch {
			case strings.Contains(row, "weather"):
				fmt.Println("Call Weather API")
			case strings.Contains(row, "food_recommendation"):
				fmt.Println("Call tourism_info API")
			case strings.Contains(row, "food_recommendation"):
				fmt.Println("Call tourism_info API")
			case strings.Contains(row, "Date="):
				re := regexp.MustCompile(`Date=(\d{4}-\d{2}-\d{2})`)
				match := re.FindStringSubmatch(row)
				if len(match) > 1 {
					dateStr := match[1]
					date, err := time.Parse("2006-01-02", dateStr)
					if err == nil {
						fmt.Println("For the date: ", date)
					}
				}
			}
		}
	}
}

func askKLLM(llm aillm.LLMContainer, query string) *llms.ContentResponse {
	log.Println("LLM Reply to " + ":")
	queryResult, err := llm.AskLLM("", llm.WithExactPrompt(query), llm.WithStreamingFunc(print))
	response := queryResult.Response
	if err != nil {
		panic(err)
	}
	log.Println("CompletionTokens: ", response.Choices[0].GenerationInfo["CompletionTokens"])
	log.Println("PromptTokens: ", response.Choices[0].GenerationInfo["PromptTokens"])
	log.Println("TotalTokens: ", response.Choices[0].GenerationInfo["TotalTokens"])
	return response
}

func print(ctx context.Context, chunk []byte) error {
	fmt.Print(string(chunk))
	return nil
}
